"""Resumable development generation and causal checks for the small ZERO model."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from boundary_common import contract, identity, write, EXPERIMENT
from boundary_data import forward
from canada_narrative import digest, read_json
from review_canada_narrative import repeat_rate
from run_canada_narrative import setup, amp
from subword_model import sample


def mechanism_check(model, tokens, device):
    model.eval(); tokens = tokens[:, :min(32, tokens.shape[1])]
    if tokens.shape[1] < 8:
        raise ValueError('Mechanism check needs at least eight input tokens')
    cut = tokens.shape[1] // 2
    segments = torch.zeros_like(tokens); segments[:, cut:] = 1
    changed = tokens.clone(); changed[:, :cut] = (changed[:, :cut] + 3) % model.vocab
    with torch.no_grad(), amp(device):
        plain = model(tokens)
        single = forward(model, tokens, torch.zeros_like(tokens), True)
        isolated = forward(model, tokens, segments, True)
        perturbed = forward(model, changed, segments, True)
        shared_changed = model(changed)
        _, cache = model.forward_cached(tokens[:, :-1])
        cached, _ = model.forward_cached(tokens[:, -1:], cache)
        future = tokens.clone(); future[:, cut:] = (future[:, cut:] + 5) % model.vocab
        future_logits = model(future)
    metrics = {
        'single_record_max_error': float((single - plain).abs().max()),
        'isolated_previous_record_effect': float((isolated[:, cut:] - perturbed[:, cut:]).abs().max()),
        'shared_previous_record_effect': float((plain[:, cut:] - shared_changed[:, cut:]).abs().max()),
        'cached_max_error': float((plain[:, -1:] - cached).abs().max()),
        'future_token_effect': float((plain[:, :cut] - future_logits[:, :cut]).abs().max())}
    tolerance = .05 if device == 'cuda' else 2e-5
    metrics['tolerance'] = tolerance
    metrics['passed'] = (all(math.isfinite(v) for v in metrics.values()) and
        metrics['single_record_max_error'] <= tolerance and metrics['cached_max_error'] <= tolerance and
        metrics['isolated_previous_record_effect'] <= 1e-6 and metrics['future_token_effect'] <= 1e-6 and
        metrics['shared_previous_record_effect'] > 1e-6)
    if not metrics['passed']:
        raise ValueError('Mechanism check failed: ' + json.dumps(metrics))
    return metrics


def prompts(data_dir, count):
    original = read_json(data_dir / 'evaluation.json')['prompts']
    # Already-opened historical prompts are explicitly development material.
    chosen = [original[i * len(original) // count]['prompt'] for i in range(count)]
    return [{'case_id': i+1, 'family': 'historical-A-validation',
             'kind': 'historical-slice', 'prompt': p} for i,p in enumerate(chosen)]


def load_rows(directory, run_identity):
    result = {}
    for path in sorted(directory.glob('case-*.json')):
        record = read_json(path); row = record['payload']
        if record['run_identity'] != run_identity or record['payload_sha256'] != identity(row):
            raise ValueError('Saved generation identity differs: ' + str(path))
        key = (row['decoder'], row['case_id'])
        if key in result:
            raise ValueError('Duplicate saved generation')
        result[key] = row
    return result


def generate(model, tokenizer, cases, settings, output, run_identity, count=256, seed=107, device='cuda'):
    run_identity = {'caller':run_identity, 'cases_sha256':identity(cases), 'settings':settings,
                    'new_tokens':count, 'seed':seed, 'device':device}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'identity.json').exists():
        if read_json(output / 'identity.json') != run_identity:
            raise ValueError('Generation resume identity differs')
    elif any(output.iterdir()):
        raise ValueError('Generation directory lacks identity')
    else:
        write(output / 'identity.json', run_identity)
    rows = load_rows(output, run_identity)
    expected = {(decoder, c['case_id']) for decoder in settings for c in cases}
    if set(rows) - expected:
        raise ValueError('Saved case outside the requested generation set')
    for decoder, spec in settings.items():
        for case in cases:
            key = (decoder, case['case_id'])
            if key in rows:
                continue
            encoded = tokenizer.encode(case['prompt']).ids
            if tokenizer.decode(encoded) != case['prompt'] or len(encoded) + count > model.context:
                raise ValueError('Prompt round trip or context budget differs')
            started = time.monotonic()
            with amp(device):
                _, tokens = sample(model, tokenizer, case['prompt'], count=count, seed=seed,
                    return_tokens=True, temperature=spec['temperature'], top_k=spec['top_k'],
                    repetition_penalty=spec['penalty'], use_cache=True)
            generated = tokens[len(encoded):]
            row = {**case, 'decoder': decoder, 'continuation': tokenizer.decode(generated),
                   'generated_tokens': generated, 'seconds': time.monotonic()-started,
                   'prompt_tokens': len(encoded), 'repetition': {
                       str(n): repeat_rate(tokenizer.decode(generated[:n])) for n in [64,128,256] if n<=count}}
            write(output / f'case-{decoder}-{case["case_id"]:03d}.json',
                  {'run_identity': run_identity, 'payload': row, 'payload_sha256': identity(row)})
            rows[key] = row
            print('Generated', decoder, case['case_id'], flush=True)
    write(output / 'summary.json', {'status':'passed', 'continuations':len(rows),
          'scope':'Development samples; coherence and factual judgments require reading',
          'mean_repetition': {d:{str(n):sum(r['repetition'][str(n)] for (dec,_),r in rows.items() if dec==d)/len(cases)
                                    for n in [64,128,256] if n<=count} for d in settings}})
    return [rows[(d,c['case_id'])] for d in settings for c in cases]


def diagnose(data_dir, checkpoint, output, decoder_names=None, extra_cases=None):
    config = contract(); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, optimizer = setup(config['seed'], device); del optimizer
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model']); model.eval()
    tokenizer = Tokenizer.from_file(str(data_dir / 'tokenizer.json'))
    cases = prompts(data_dir, config['generation']['cases'])
    if extra_cases:
        extra = read_json(extra_cases)
        if any(set(c) != {'case_id','family','kind','prompt'} for c in extra):
            raise ValueError('Extra cases must contain input-only fields')
        cases += extra
    ids = [c['case_id'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Prompt case IDs must be unique')
    settings = {k:v for k,v in config['decoders'].items() if decoder_names is None or k in decoder_names}
    if not settings or decoder_names and set(decoder_names) != set(settings):
        raise ValueError('Choose registered decoders')
    run_identity = {'checkpoint_sha256':digest(checkpoint), 'tokenizer_sha256':digest(data_dir/'tokenizer.json'),
                    'cases_sha256':identity(cases), 'settings':settings, 'generation':config['generation'],
                    'device':device, 'contract_sha256':identity(config),
                    'source_lock_sha256':digest(EXPERIMENT/'source.lock.json')}
    tokens = torch.tensor([tokenizer.encode(cases[0]['prompt']).ids], device=device)
    checks = mechanism_check(model, tokens, device)
    # Development prompt loss is reported per case and family, with no claim
    # that a template variant is an independent source or author.
    prompt_scores = []
    token_counts = [0] * model.vocab
    for case in cases:
        encoded = tokenizer.encode(case['prompt']).ids
        for token in encoded:
            token_counts[token] += 1
        with torch.no_grad(), amp(device):
            x = torch.tensor([encoded[:-1]], device=device)
            logits = model(x).float()
            target = torch.tensor(encoded[1:], device=device)
            loss = torch.nn.functional.cross_entropy(logits[0], target, reduction='sum')
        byte_count = sum(len(tokenizer.decode([t]).encode()) for t in encoded[1:])
        prompt_scores.append({'case_id': case['case_id'], 'family': case['family'],
                              'target_tokens': len(encoded)-1,
                              'bits_per_decoded_token_bytes': float(loss) / math.log(2) / max(1,byte_count)})
    names = ['Mara','Tomas','Janet','Bridget','Rose','Daniel','Alice','Peter']
    token_audit = {'scope': 'Development prompts only', 'token_counts': token_counts,
                   'name_tokenization': {n:tokenizer.encode(n).ids for n in names},
                   'prompt_scores': prompt_scores}
    rows = generate(model, tokenizer, cases, settings, output, run_identity,
                    config['generation']['new_tokens'], config['generation']['seed'], device)
    write(output / 'mechanism-check.json', checks)
    write(output / 'token-audit.json', token_audit)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data','checkpoint','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--decoder', action='append')
    parser.add_argument('--extra-cases', type=Path)
    a=parser.parse_args(); diagnose(a.data,a.checkpoint,a.output,a.decoder,a.extra_cases)
