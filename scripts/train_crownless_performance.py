"""Train and evaluate the matched three-creature, three-emotion pilot locally."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import torch
from tokenizers import Tokenizer
from crownless_conversation import build_rows
from crownless_performance import VERSION, corpus, score, selected
from crownless_v2 import batch, encode_row, generate, save
from crownless_v2_export import load_export, export


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n')
def encoded(tokenizer, row): return encode_row(tokenizer, row, slots=True, conversation=True)


def evaluate(model, tokenizer, rows, rules):
    model.to('cpu').eval()
    results = []
    for row in rows:
        result = generate(model, tokenizer, encoded(tokenizer, row))
        results.append({'id': row['id'], 'rule': row['rule'], 'act': row['act'],
                        'performance': row.get('performance'), 'confidence': row['confidence'],
                        'retold': row['retold'], 'reference': row['output'],
                        'history': row.get('history', []), 'text': result['text'],
                        'stopped': result['stopped'], **score(row, rules[row['rule']], result)})
    def counts(values):
        return {'count': len(values), **{key: sum(bool(r[key]) for r in values)
                for key in ('meaning', 'identity', 'emotion', 'joint', 'stopped')}}
    groups = {}
    for row in results:
        key = ':'.join(row['performance'].values()) if row['performance'] else 'ordinary'
        groups.setdefault(key, []).append(row)
    return {**counts(results), 'groups': {k: counts(v) for k, v in groups.items()}, 'rows': results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('models/crownless-conversation/core.ccv2'))
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=91)
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    if args.steps < 1: p.error('Steps must be positive')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, metadata = load_export(args.base, args.tokenizer)
    if model.mode != 'conversation': p.error('Use the conversation core')
    if sha(args.data / 'rules.json') != metadata['rules_sha256']: p.error('Grammar differs')
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    bases = {s: [json.loads(x) for x in (args.data / f'{s}.jsonl').read_text().splitlines()]
             for s in ('train', 'validation', 'test')}
    for values in bases.values():
        for row in values: row['kind_id'] = metadata['meaning_ids'][row['rule']]
    # Reserve whole event kinds for style transfer; the base model has learned
    # these meanings already. This measures new style/meaning combinations.
    kinds = sorted({r['kind'] for r in bases['train']})
    held_kinds = kinds[::9]
    held = sorted({r['rule'] for r in bases['train'] if r['kind'] in held_kinds})
    rows = {
        'train': corpus(bases['train'], rules, args.seed, 1200, held),
        'rehearsal': build_rows(selected(bases['train'], 1200), rules, args.seed),
        'validation': corpus(bases['validation'], rules, args.seed+1, 36, held),
        'test': corpus(bases['test'], rules, args.seed+2, 72, held),
        'new_questions': corpus(bases['test'], rules, args.seed+3, 36, held, paraphrase=True),
        'new_events': corpus([r for r in bases['test'] if r['rule'] in held], rules, args.seed+4, 24),
        'retention': build_rows(selected(bases['test'], 122), rules, args.seed+5),
    }
    args.output.mkdir(parents=True)
    files = {}
    for split, values in rows.items():
        path = args.output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in values))
        files[split] = {'rows': len(values), 'sha256': sha(path)}
    info = {'schema': VERSION, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                ('crownless_performance.py', 'train_crownless_performance.py', 'crownless_v2.py',
                 'crownless_conversation.py', 'crownless_v2_export.py', 'score_crownless_v2.py')},
            'base_sha256': sha(args.base), 'tokenizer_sha256': sha(args.tokenizer),
            'data_manifest_sha256': sha(args.data / 'manifest.json'), 'grammar_sha256': metadata['rules_sha256'],
            'seed': args.seed, 'steps': args.steps, 'device': args.device, 'batch_size': 16,
            'rehearsal_fraction': .5, 'held_event_kinds': held_kinds, 'held_rules': held, 'files': files,
            'parameters': sum(p.numel() for p in model.parameters()),
            'torch': torch.__version__, 'scope': 'Authored style forms; exact meaning and style scored separately.'}
    write(args.output / 'manifest.json', info)
    baseline = {}
    for split in ('test', 'retention'):
        baseline[split] = evaluate(model, tokenizer, rows[split], rules)
        write(args.output / f'base-{split}.json', baseline[split])
        print(json.dumps({'base': split, **{k:v for k,v in baseline[split].items() if k not in ('rows','groups')}}), flush=True)
    model.to(args.device)
    training = [encoded(tokenizer, r) for r in rows['train']]
    rehearsal = [encoded(tokenizer, r) for r in rows['rehearsal']]
    validation = [encoded(tokenizer, r) for r in rows['validation']]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    rng = random.Random(args.seed); best = float('inf'); history = []; start = time.monotonic()
    for step in range(1, args.steps+1):
        model.train()
        inputs = batch(rng.choices(training, k=8) + rng.choices(rehearsal, k=8), args.device)
        for group in optimizer.param_groups:
            group['lr'] = 1e-4 * min(step/50, 1) * (.1 + .9*(1-step/args.steps))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                values = [model.loss(batch(validation[i:i+16], args.device)).item()
                          for i in range(0, len(validation), 16)]
            val = sum(values)/len(values)
            item = {'step': step, 'loss': loss.item(), 'validation': val, 'seconds': time.monotonic()-start}
            history.append(item); write(args.output / 'history.json', history)
            print(json.dumps(item), flush=True)
            if val < best:
                best = val
                save(args.output / 'best.pt', model, args.tokenizer,
                     {k:metadata[k] for k in ('meaning_ids','kind_ids','rules_sha256')} | {'step':step})
    saved = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.to('cpu').load_state_dict(saved['state'])
    export(model, args.tokenizer, args.output / 'core.ccv2', saved)
    compact, _ = load_export(args.output / 'core.ccv2', args.tokenizer)
    reports = {}
    for split in ('test','new_questions','new_events','retention'):
        result = evaluate(compact, tokenizer, rows[split], rules)
        write(args.output / f'{split}-results.json', result)
        reports[split] = {k:v for k,v in result.items() if k != 'rows'}
        print(json.dumps({split: reports[split]}), flush=True)
    reports['base'] = {s:{k:v for k,v in r.items() if k != 'rows'} for s,r in baseline.items()}
    reports['model_sha256'] = sha(args.output / 'core.ccv2')
    reports['model_bytes'] = (args.output / 'core.ccv2').stat().st_size
    reports['selected_step'] = saved['step']
    write(args.output / 'results.json', reports)


if __name__ == '__main__': main()
