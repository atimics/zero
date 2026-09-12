"""Evaluate saved float or compact weights on the frozen paired corpus."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from crownless_v2 import encode_row, load
from crownless_v2_export import load_export
from train_crownless_v2 import evaluate, read
from score_crownless_v2 import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokenizer', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--limit', type=int, default=1250)
    args = p.parse_args()
    if args.output.exists(): p.error('Choose a fresh output directory')
    if args.limit < 1: p.error('Use a positive row limit')
    torch.set_num_threads(4)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    model, metadata = (load_export if args.model.suffix == '.ccv2' else load)(args.model, args.tokenizer, args.device)
    if metadata['rules_sha256'] != digest(args.data / 'rules.json'): p.error('Model and data grammar differ')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output.mkdir(parents=True)
    (args.output / 'identities.json').write_text(json.dumps({
        'model_sha256': digest(args.model), 'tokenizer_sha256': digest(args.tokenizer),
        'data_manifest_sha256': digest(args.data / 'manifest.json'),
        'device': args.device, 'requested_rows': args.limit}, indent=2) + '\n')
    for split in ('test', 'wording'):
        rows = read(args.data / f'{split}.jsonl')[:args.limit]
        for row in rows:
            row['kind_id'] = metadata['meaning_ids'][row['rule']] if model.mode == 'packet' else metadata['kind_ids'][row['kind']]
        records = [encode_row(tokenizer, row, model.config.context,
                   slots=model.mode in ('slots', 'packet'), packet=model.mode == 'packet') for row in rows]
        result = evaluate(model, tokenizer, records, args.device, args.limit)
        (args.output / f'{split}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'split': split, **{k: v for k, v in result.items() if k != 'rows'}}), flush=True)
    report = score(args.data, args.output)
    (args.output / 'meaning.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({s: {k: v for k, v in r.items() if k not in ('rows', 'kinds')}
                     for s, r in report['splits'].items()}), flush=True)


if __name__ == '__main__': main()
