"""Run matched text, field, and copy pilots with saved data and model identities."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import torch
from crownless_v2 import Crownless, Config, batch, encode_row, generate, save, train_tokenizer
from tokenizers import Tokenizer


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, tokenizer, records, device, limit):
    model.eval()
    selected = records[:limit]
    outputs = []
    for record in selected:
        row = record['row']
        result = generate(model, tokenizer, record, device)
        required = sorted(set(s['text'] for s in row['copies']))
        outputs.append({'id': row['id'], 'pair': row['pair'], 'kind': row['kind'],
                        'prefix': row['prefix'], 'reference': row['output'],
                        'text': result['text'], 'stopped': result['stopped'],
                        'exact_reference': result['text'] == row['output'],
                        'required_names': required,
                        'all_names': all(name in result['text'] for name in required),
                        'copy_actions': sum('copy' in a for a in result['actions'])})
    return {'rows': outputs, 'count': len(outputs),
            'exact_reference': sum(r['exact_reference'] for r in outputs),
            'all_names': sum(r['all_names'] for r in outputs),
            'stopped': sum(r['stopped'] for r in outputs),
            'scope': 'Reference match and required-name presence; independent full-fidelity review remains separate.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default='mps')
    p.add_argument('--eval-rows', type=int, default=84)
    p.add_argument('--modes', nargs='+', choices=['text', 'fields', 'copy'], default=['text', 'fields', 'copy'])
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--kind-features', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a fresh output directory')
    if min(args.steps, args.batch_size, args.eval_rows) < 1: p.error('Use positive counts')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    rows = {s: read(args.data / f'{s}.jsonl') for s in ('train', 'validation', 'test', 'wording')}
    manifest = json.loads((args.data / 'manifest.json').read_text())
    kind_ids = {r['kind']: r['value'] + 1 for r in manifest['coverage']['events']}
    for split in rows.values():
        for row in split: row['kind_id'] = kind_ids[row['kind']]
    tokenizer_path = args.output / 'tokenizer.json'
    if args.tokenizer:
        tokenizer_path.write_bytes(args.tokenizer.read_bytes())
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = train_tokenizer(rows['train'], tokenizer_path)
    records = {s: [encode_row(tokenizer, row) for row in values] for s, values in rows.items()}
    metadata = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'sources': {name: sha(Path(__file__).with_name(name)) for name in
                            ('crownless_v2.py', 'train_crownless_v2.py')},
                'data_manifest_sha256': sha(args.data / 'manifest.json'),
                'tokenizer_sha256': sha(tokenizer_path), 'seed': args.seed,
                'steps': args.steps, 'batch_size': args.batch_size, 'device': args.device,
                'torch': torch.__version__, 'english_replay': 0,
                'kind_ids': kind_ids, 'kind_features': args.kind_features,
                'evaluation_scope': 'Synthetic shared-rule diagnostic; fresh weights, one seed per arm.',
                'target_tokens': sum(sum(x >= 0 for x in r['labels']) for r in records['train']),
                'max_sequence': max(len(r['tokens']) for split in records.values() for r in split)}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    summary = {}
    for mode in args.modes:
        torch.manual_seed(args.seed)
        config = Config(kinds=max(kind_ids.values()) + 1 if args.kind_features else 0)
        model = Crownless(config, mode).to(args.device)
        parameters = sum(p.numel() for p in model.parameters())
        assert parameters == 4924033 + config.kinds * config.dim and parameters <= 5000000
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.01)
        rng = random.Random(args.seed)
        directory = args.output / mode
        directory.mkdir()
        best, history, started = float('inf'), [], time.monotonic()
        for step in range(1, args.steps + 1):
            model.train()
            selected = [records['train'][rng.randrange(len(records['train']))] for _ in range(args.batch_size)]
            inputs = batch(selected, args.device)
            lr = 4e-4 * min(step / 100., 1.) * (.1 + .9 * (1 - step / args.steps))
            for group in optimizer.param_groups: group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs)
            if not torch.isfinite(loss): raise ValueError('Non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % 250 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    val = sum(model.loss(batch(records['validation'][i:i+16], args.device)).item()
                              for i in range(0, min(128, len(records['validation'])), 16)) / 8
                item = {'step': step, 'train_loss': loss.item(), 'validation_loss': val,
                        'seconds': time.monotonic() - started}
                history.append(item)
                (directory / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
                print(json.dumps({'mode': mode, **item}), flush=True)
                if val < best:
                    best = val
                    save(directory / 'best.pt', model, tokenizer_path, {'step': step, 'seed': args.seed})
        model.load_state_dict(torch.load(directory / 'best.pt', map_location=args.device, weights_only=True)['state'])
        result = {'parameters': parameters, 'training_seconds': time.monotonic() - started, 'splits': {}}
        for split in ('test', 'wording'):
            scored = evaluate(model, tokenizer, records[split], args.device, args.eval_rows)
            (directory / f'{split}.json').write_text(json.dumps(scored, indent=2) + '\n')
            result['splits'][split] = {k: v for k, v in scored.items() if k != 'rows'}
        summary[mode] = result
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'mode': mode, 'result': result}), flush=True)


if __name__ == '__main__': main()
