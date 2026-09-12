"""Check changed prior speech and retention of single-account wording."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from tokenizers import Tokenizer
from crownless_v2_export import load_export
from crownless_conversation import response
from train_crownless_conversation import evaluate, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokenizer', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.set_num_threads(4)
    model, metadata = load_export(args.model, args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata['rules_sha256'] != sha(args.data / 'rules.json'): p.error('Grammar differs')
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    bases = read(args.data / 'test.jsonl')
    for row in bases: row['kind_id'] = metadata['meaning_ids'][row['rule']]
    pairs = []
    for i, base in enumerate(bases[:122]):
        for act in ('agree', 'disagree'):
            row = response(base, rules[base['rule']], act, random.Random(1000 + i))
            row['history'] = [row['history'][-1]]
            row['id'] += ':strict:' + act
            pairs.append(row)
    controls = [{**row, 'history': []} for row in pairs]
    retained = [{**r, 'act': 'start', 'history': []} for r in bases]
    args.output.mkdir(parents=True)
    summary = {'model_sha256': sha(args.model), 'tokenizer_sha256': sha(args.tokenizer),
               'data_manifest_sha256': sha(args.data / 'manifest.json'), 'tests': {}}
    for name, rows in [('changed_history', pairs), ('history_removed', controls), ('account_retention', retained)]:
        report = evaluate(model, tokenizer, rows, rules)
        (args.output / (name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        summary['tests'][name] = {k:v for k,v in report.items() if k != 'rows'}
        if name == 'changed_history':
            changed, passed, response_changes = 0, 0, 0
            for i in range(0, len(rows), 2):
                if rows[i+1]['changed'] is None: continue
                changed += 1
                a, b = report['rows'][i:i+2]
                passed += a['approved'] and b['approved']
                response_changes += a['text'] != b['text']
            summary['history_pairs'] = {'changed_pairs': changed, 'both_responses_approved': passed,
                                        'response_changed': response_changes}
        print(json.dumps({name: summary['tests'][name]}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__': main()
