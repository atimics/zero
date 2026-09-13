"""Score a finished pair and create the existing five-pair review page."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from canada_narrative import EXPERIMENT, ROOT, digest, read_json, write_json, verify_prepared
from ab_review import effective_votes, validate_vote


def repeat_rate(text):
    words = text.lower().split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.


def validate_pair(control, candidate, comparison, manifest_sha):
    contract = read_json(EXPERIMENT / 'contract.json')
    spec = contract['comparisons'][comparison]
    for record, arm in zip([control, candidate], spec['arms']):
        if (record['status'] != 'trained' or record['arm'] != arm or
                record['comparison'] != comparison or record['seed'] not in contract['seeds'] or
                record['target_presentations'] != spec['target_tokens_per_arm'] or
                record['contract_sha256'] != digest(EXPERIMENT / 'contract.json') or
                record['implementation_sha256'] != digest(EXPERIMENT / 'implementation.lock.json') or
                record['manifest_sha256'] != manifest_sha):
            raise ValueError('Completed run differs from the registered pair')
    for name in ['seed', 'initial_weights_sha256', 'config', 'parameters', 'device', 'platform']:
        if control[name] != candidate[name]:
            raise ValueError(f'Paired run mismatch: {name}')


def build(args):
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from run_canada_narrative import evaluate, setup, amp, verify_code
    from subword_model import sample
    verify_code(); verify_prepared(args.data)
    records = [read_json(directory / 'result.json') for directory in [args.control, args.candidate]]
    validate_pair(*records, args.comparison, digest(args.data / 'manifest.json'))
    if digest(args.data / 'manifest.json') != read_json(EXPERIMENT / 'preparation.json')['manifest_sha256']:
        raise ValueError('Prepared input receipt mismatch')
    for directory, result in zip([args.control, args.candidate], records):
        if digest(directory / 'best.pt') != result['best_sha256']:
            raise ValueError('Selected checkpoint identity mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation = read_json(args.data / 'evaluation.json')
    data = np.memmap(args.data / 'validation.bin', dtype='<u2', mode='r')
    lengths = np.array(read_json(args.data / 'token_bytes.json'))
    tokenizer = Tokenizer.from_file(str(args.data / 'tokenizer.json'))
    scores = []; samples = []; rates = []
    for directory, result in zip([args.control, args.candidate], records):
        device = result['device']; model, optimizer = setup(result['seed'], device)
        del optimizer
        state = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
        if state['step'] != result['selected_step'] or state['config'] != result['config']:
            raise ValueError('Checkpoint selection identity mismatch')
        model.load_state_dict(state['model'])
        scores.append(evaluate(model, data, evaluation['outcome'], lengths, device))
        outputs = []
        for row in evaluation['prompts']:
            with amp(device):
                full = sample(model, tokenizer, row['prompt'], count=256, seed=107,
                              repetition_penalty=1.1, temperature=.7, top_k=40, use_cache=True)
            outputs.append(full[len(row['prompt']):])
        samples.append(outputs); rates.append(sum(map(repeat_rate, outputs)) / len(outputs))
        del model, state
    cases = []; key = []
    # Exactly 100 cases per orientation, with a fixed hash-based shuffle.
    order = sorted(range(200), key=lambda i: hashlib.sha256(f'canada-v1:{i}'.encode()).digest())
    candidate_left = set(order[:100])
    for i, prompt in enumerate(evaluation['prompts']):
        left, right = (1, 0) if i in candidate_left else (0, 1)
        cases.append({'case_id': i + 1, 'prompt': prompt['prompt'],
                      'A': samples[left][i], 'B': samples[right][i]})
        key.append({'case_id': i + 1, 'candidate': 'A' if left == 1 else 'B'})
    packet = {'schema_version': 1, 'packet_id': 'zero-canada-' + hashlib.sha256(
        json.dumps(cases, sort_keys=True).encode()).hexdigest()[:16], 'cases': cases}
    write_json(args.output / 'ab-packet.json', packet)
    write_json(args.output / 'blind-key.json', key)
    template = (ROOT / 'scripts/ab_review.html').read_text()
    rubric = read_json(EXPERIMENT / 'contract.json')['evaluation']['human_review']['rubric']
    template = template.replace('<p>Pick A or B. Skip whenever you want.</p>',
                                '<p>' + html.escape(rubric) + '</p>')
    payload = json.dumps(packet).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (args.output / 'ab-review.html').write_text(template.replace('/*PACKET*/null', payload))
    improvement = 1 - scores[1]['bits_per_byte'] / scores[0]['bits_per_byte']
    write_json(args.output / 'metrics.json', {'comparison': args.comparison, 'runs': records,
        'contract_sha256': digest(EXPERIMENT / 'contract.json'),
        'scores': scores, 'relative_improvement': improvement,
        'repetition_rates': rates, 'numerical_gates_pass': improvement >= .02 and rates[1] <= rates[0] + .01,
        'packet_sha256': digest(args.output / 'ab-packet.json'), 'key_sha256': digest(args.output / 'blind-key.json')})


def wilson(wins, count):
    if count == 0:
        return None
    z = 1.96; p = wins / count; denominator = 1 + z*z / count
    center = (p + z*z / (2*count)) / denominator
    width = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return [center-width, center+width]


def tally(events, packet, key, reviewer):
    votes = effective_votes([validate_vote(event, packet) for event in events])
    votes = [v for v in votes if v['reviewer_id'] == reviewer]
    mapping = {r['case_id']: r['candidate'] for r in key}
    if set(mapping) != {r['case_id'] for r in packet['cases']}:
        raise ValueError('Review key case mismatch')
    wins = sum(v['choice'] == mapping[v['case_id']] for v in votes)
    skips = sum(v['choice'] == 'skip' for v in votes)
    decisive = len(votes) - skips
    return {'reviewer': reviewer, 'reviewed': len(votes), 'candidate_wins': wins,
            'control_wins': decisive-wins, 'skips': skips,
            'candidate_share_all_200': wins / 200,
            'decisive_wilson_95': wilson(wins, decisive),
            'human_gate_pass': len(votes) == 200 and decisive >= 160 and wins >= 120,
            'scope': 'Descriptive result for one reviewer and these prompts; skip includes ties and unclear choices'}


def summarize(args):
    packet = read_json(args.directory / 'ab-packet.json'); key = read_json(args.directory / 'blind-key.json')
    metrics = read_json(args.directory / 'metrics.json')
    if (digest(args.directory / 'ab-packet.json') != metrics['packet_sha256'] or
            digest(args.directory / 'blind-key.json') != metrics['key_sha256'] or
            metrics['contract_sha256'] != digest(EXPERIMENT / 'contract.json')):
        raise ValueError('Review evidence identity mismatch')
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    result = tally(events, packet, key, args.reviewer)
    result.update({'comparison': metrics['comparison'],
                   'seed': metrics['runs'][0]['seed'],
                   'contract_sha256': metrics['contract_sha256'],
                   'manifest_sha256': metrics['runs'][0]['manifest_sha256'],
                   'numerical_gates_pass': metrics['numerical_gates_pass'],
                   'pilot_pass': metrics['numerical_gates_pass'] and result['human_gate_pass']})
    write_json(args.output, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build')
    for name in ['data', 'control', 'candidate', 'output']:
        build_parser.add_argument('--' + name, type=Path, required=True)
    build_parser.add_argument('--comparison', choices=['AB', 'BC'], required=True)
    summary = sub.add_parser('summarize')
    for name in ['directory', 'events', 'output']:
        summary.add_argument('--' + name, type=Path, required=True)
    summary.add_argument('--reviewer', required=True)
    args = parser.parse_args()
    (build if args.command == 'build' else summarize)(args)
