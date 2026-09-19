"""Build the thirteen-act Crownless corpus: nine conversation acts, four mind acts.

The conversation acts came back from zero@7cf61c6d, where they were replaced
rather than extended when the mind work landed. A model serving the runtime has
to answer both entry points -- CcCoreModelBegin asks for a conversation turn
with no mind context, CcCoreModelBeginMind for a mind turn with one -- so a
corpus that covers only one family costs the other.

Every split is a pure function of the account corpus and the seed, and the
manifest records the hashes needed to reproduce or verify it byte for byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crownless_conversation import build_rows, ACTS, CONVERSATION_ACTS, MIND_ACTS


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]


def report(rows):
    acts = Counter(r['act'] for r in rows)
    return {'rows': len(rows),
            'acts': dict(sorted(acts.items())),
            'with_mind': sum(1 for r in rows if r.get('mind')),
            'kinds': len({r['kind'] for r in rows}),
            'history_depth': dict(sorted(Counter(len(r['history']) for r in rows).items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accounts', type=Path, required=True,
                   help='Account corpus: conversation acts, full event-kind coverage')
    p.add_argument('--mind', type=Path, required=True,
                   help='Sim-derived mind corpus: mind acts, real goal and stress')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')

    rules = {r['id']: r for r in json.loads((args.accounts / 'rules.json').read_text())['rules']}
    if sha(args.accounts / 'rules.json') != sha(args.mind / 'rules.json'):
        p.error('The two corpora were built from different grammars')
    accounts = {s: read(args.accounts / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    minds = {s: read(args.mind / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    args.output.mkdir(parents=True)
    (args.output / 'rules.json').write_bytes((args.accounts / 'rules.json').read_bytes())

    splits = {}
    for split in ('train', 'validation', 'test'):
        repeats = args.repeats if split == 'train' else 1
        splits[split] = (
            build_rows(accounts[split], rules, f'{args.seed}:{split}:conversation',
                       repeats=repeats, acts=CONVERSATION_ACTS) +
            build_rows(minds[split], rules, f'{args.seed}:{split}:mind',
                       repeats=repeats, acts=MIND_ACTS))
    # Reserved paraphrases of the prompts, scored apart from the trained wording.
    splits['wording'] = (
        build_rows(accounts['test'], rules, f'{args.seed}:wording:conversation',
                   paraphrase=True, acts=CONVERSATION_ACTS) +
        build_rows(minds['test'], rules, f'{args.seed}:wording:mind',
                   paraphrase=True, acts=MIND_ACTS))

    files = {}
    for split, rows in splits.items():
        path = args.output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        files[split] = {'sha256': sha(path), **report(rows)}

    manifest = {
        'schema': 'crownless.core_acts.v1',
        'seed': args.seed, 'repeats': args.repeats,
        'acts': {'conversation': CONVERSATION_ACTS, 'mind': MIND_ACTS, 'total': len(ACTS)},
        'accounts_manifest_sha256': sha(args.accounts / 'manifest.json'),
        'mind_manifest_sha256': sha(args.mind / 'manifest.json'),
        'rules_sha256': sha(args.accounts / 'rules.json'),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                          ('crownless_conversation.py', 'crownless_v2.py', 'build_crownless_acts.py')},
        'recovered_from': {'commit': '7cf61c6df77f83bb259820264dc8b5548d34198e',
                           'note': 'Conversation acts restored from the pre-mind generator.'},
        'splits': files,
        'scope': 'Conversation turns answered from the account alone and mind turns answered '
                 'under goal, stress, courage, memories and thoughts, over shared account rows.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({s: {'rows': v['rows'], 'acts': len(v['acts']), 'with_mind': v['with_mind']}
                      for s, v in files.items()}, indent=2))


if __name__ == '__main__': main()
