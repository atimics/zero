"""Score unused account pairs after the pilot development checks."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_performance import corpus
from crownless_v2_export import load_export
from train_crownless_performance import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.set_num_threads(4)
    model, meta = load_export(args.pilot/'core.ccv2', args.tokenizer)
    if hashlib.sha256((args.data/'rules.json').read_bytes()).hexdigest() != meta['rules_sha256']:
        p.error('Grammar differs')
    seen = set()
    for split in ('test','new_questions','new_events','retention'):
        seen.update(json.loads(s)['pair'] for s in (args.pilot/f'{split}.jsonl').read_text().splitlines())
    bases = [json.loads(s) for s in (args.data/'test.jsonl').read_text().splitlines()]
    bases = [r for r in bases if r['pair'] not in seen]
    for row in bases: row['kind_id'] = meta['meaning_ids'][row['rule']]
    rules = {r['id']:r for r in json.loads((args.data/'rules.json').read_text())['rules']}
    rows = corpus(bases, rules, 20260923, 72, paraphrase=True)
    if len(rows) != 648: p.error('Need 72 fresh account rows')
    args.output.mkdir(parents=True)
    path = args.output/'rows.jsonl'
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    result = evaluate(model, Tokenizer.from_file(str(args.tokenizer)), rows, rules)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['rows_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['model_sha256'] = hashlib.sha256((args.pilot/'core.ccv2').read_bytes()).hexdigest()
    result['scope'] = 'Fresh account pairs after development evaluation; seed fixed at 20260923, fresh question wording.'
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','groups')}))


if __name__ == '__main__': main()
