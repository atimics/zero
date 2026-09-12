"""Tune the 5M core on actual preceding speech and held-account responses."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import torch
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row, generate, save
from crownless_v2_export import load_export, export
from crownless_conversation import build_rows, approved_response, response


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def encoded(tokenizer, row): return encode_row(tokenizer, row, slots=True, conversation=True)


def evaluate(model, tokenizer, rows, rules):
    model = model.to('cpu').eval()
    scored = []
    for row in rows:
        result = generate(model, tokenizer, encoded(tokenizer, row))
        scored.append({'id': row['id'], 'rule': row['rule'], 'act': row['act'],
                       'history': row['history'], 'reference': row['output'],
                       'text': result['text'], 'stopped': result['stopped'],
                       'approved': result['stopped'] and result['text'] in approved_response(row, rules[row['rule']])})
    return {'count': len(scored), 'approved': sum(x['approved'] for x in scored),
            'stopped': sum(x['stopped'] for x in scored),
            'acts': {act: {'count': sum(x['act'] == act for x in scored),
                          'approved': sum(x['act'] == act and x['approved'] for x in scored)}
                     for act in sorted({x['act'] for x in scored})}, 'rows': scored}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('models/crownless-core-v2/core.ccv2'))
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    model, metadata = load_export(args.base, args.tokenizer, args.device)
    model.mode = 'conversation'
    assert sum(x.numel() for x in model.parameters()) <= 5000000
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    if sha(args.data / 'rules.json') != metadata['rules_sha256']: p.error('Grammar differs')
    bases = {s: read(args.data / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    for values in bases.values():
        for row in values: row['kind_id'] = metadata['meaning_ids'][row['rule']]
    rows = {s: build_rows(values, rules, f'{args.seed}:{s}', repeats=2 if s == 'train' else 1)
            for s, values in bases.items()}
    rows['wording'] = build_rows(bases['test'], rules, f'{args.seed}:wording', paraphrase=True)
    args.output.mkdir(parents=True)
    (args.output / 'tokenizer.json').write_bytes(args.tokenizer.read_bytes())
    files = {}
    for split, values in rows.items():
        path = args.output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in values))
        files[split] = {'rows': len(values), 'sha256': sha(path)}
    info = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                 ['crownless_v2.py', 'crownless_conversation.py', 'train_crownless_conversation.py']},
            'base_sha256': sha(args.base), 'tokenizer_sha256': sha(args.tokenizer),
            'data_manifest_sha256': sha(args.data / 'manifest.json'), 'seed': args.seed,
            'steps': args.steps, 'batch_size': 16, 'files': files,
            'scope': 'Authored response families over held-out account names; new question wording scored separately.'}
    (args.output / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    training = [encoded(tokenizer, r) for r in rows['train']]
    validation = [encoded(tokenizer, r) for r in rows['validation'][:244]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    rng = random.Random(args.seed); best = float('inf'); history = []; start = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        inputs = batch(rng.choices(training, k=16), args.device)
        for group in optimizer.param_groups:
            group['lr'] = 2e-4 * min(step / 100, 1) * (.1 + .9 * (1 - step / args.steps))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        if not torch.isfinite(loss): raise ValueError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                values = [model.loss(batch(validation[i:i+16], args.device)).item() for i in range(0, len(validation), 16)]
            val = sum(values) / len(values)
            item = {'step': step, 'loss': loss.item(), 'validation': val, 'seconds': time.monotonic()-start}
            history.append(item); print(json.dumps(item), flush=True)
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            if val < best:
                best = val
                save(args.output / 'best.pt', model, args.tokenizer,
                     {k: metadata[k] for k in ['meaning_ids', 'kind_ids', 'rules_sha256']} | {'step': step, 'seed': args.seed})
    saved = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.to('cpu').load_state_dict(saved['state'])
    export(model, args.tokenizer, args.output / 'core.ccv2', saved)
    compact, _ = load_export(args.output / 'core.ccv2', args.tokenizer)
    reports = {}
    for split in ('test', 'wording'):
        result = evaluate(compact, tokenizer, rows[split], rules)
        (args.output / f'{split}-results.json').write_text(json.dumps(result, indent=2) + '\n')
        reports[split] = {k:v for k,v in result.items() if k != 'rows'}
        print(json.dumps({split: reports[split]}), flush=True)
    # A fixed held account with only the previous statement changed.
    paired = []
    rng = random.Random(902)
    for base in bases['test'][:122]:
        for act in ('agree', 'disagree'):
            row = response(base, rules[base['rule']], act, rng)
            row['id'] += ':' + act
            paired.append(row)
    result = evaluate(compact, tokenizer, paired, rules)
    (args.output / 'history-pairs-results.json').write_text(json.dumps(result, indent=2) + '\n')
    reports['history_pairs'] = {k:v for k,v in result.items() if k != 'rows'}
    (args.output / 'results.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__': main()
