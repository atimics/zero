"""Per-move breakdown for a move-axis checkpoint.

A total says a run fell short; it does not say whether the model performs
eleven moves and fumbles one, or performs all twelve at about the same middling
rate. This scores a larger balanced sample than the training guard can afford
and, for every reply that is not a known wording of the cued move, asks the
question that decides what to do next: is it a known wording of some *other*
move, or a wording no part of the corpus contains?

The first failure is a targeting problem -- the cue is not reaching the model,
or two moves are not separable from the context they share. The second is a
fluency problem, and more steps are a plausible answer to it. They call for
different work, so they are counted separately.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer
from crownless_v2_export import load_export
from train_crownless_moves import evaluate, read, shape, shapes_by_move, vocabulary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--rows-per-move', type=int, default=20)
    p.add_argument('--examples', type=int, default=2)
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    model, metadata = load_export(args.model, args.tokenizer, 'cpu')
    model.mode = 'conversation'
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.corpus / 'rules.json').read_text())['rules']}
    meanings = metadata['meaning_ids']

    splits = {name: [{**row, 'kind_id': meanings[row['rule']]}
                     for row in read(args.corpus / f'{name}.jsonl')]
              for name in ('validation', 'test', 'wording')}
    known = vocabulary(*splits.values())
    moves = shapes_by_move(*splits.values())

    by_move = defaultdict(list)
    for row in splits['validation']:
        if len(by_move[row['move']]) < args.rows_per_move:
            by_move[row['move']].append(row)
    sample = [row for move in sorted(by_move) for row in by_move[move]]
    report, scored = evaluate(model, tokenizer, sample, rules, 'diagnose', known, moves)

    rows = {row['id']: row for row in sample}
    per_move, examples = {}, defaultdict(list)
    for move in sorted(by_move):
        scores = [x for x in scored if rows[x['id']]['move'] == move]
        elsewhere = 0
        for x in scores:
            if x['move']: continue
            written = shape(x['text'], rows[x['id']])
            other = sorted(name for name, shapes in moves.items()
                           if name != move and written in shapes)
            if other:
                elsewhere += 1
                if len(examples[move]) < args.examples:
                    examples[move].append({'reads_as': other, 'text': x['text'],
                                           'wanted': x['reference']})
            elif len(examples[move]) < args.examples:
                examples[move].append({'reads_as': ['no known wording'], 'text': x['text'],
                                       'wanted': x['reference']})
        per_move[move] = {'rows': len(scores),
                          'move': sum(x['move'] for x in scores),
                          'cell': sum(x['exact'] for x in scores),
                          'another_move': elsewhere,
                          'unknown_wording': sum(1 for x in scores if not x['move']) - elsewhere,
                          'degenerate': sum(x['degenerate'] for x in scores),
                          'distinct': len({x['text'] for x in scores})}

    summary = {'model': str(args.model), 'rows': len(sample), 'totals': report,
               'per_move': per_move, 'examples': dict(examples),
               # Every reply, so a scoring question can be re-asked of the same
               # generations instead of spending another evaluation pass.
               'scored': [{'id': x['id'], 'move': rows[x['id']]['move'], 'text': x['text'],
                           'reference': x['reference'], 'accepted': rows[x['id']].get('accepted', []),
                           'matched_move': x['move'], 'matched_cell': x['exact']} for x in scored]}
    if args.output: args.output.write_text(json.dumps(summary, indent=2) + '\n')

    width = max(len(m) for m in per_move)
    print(f'{"move":<{width}}  rows  move  cell  other  unknown  degen  distinct')
    for move, counts in sorted(per_move.items(), key=lambda kv: kv[1]['move']):
        print(f'{move:<{width}}  {counts["rows"]:>4}  {counts["move"]:>4}  {counts["cell"]:>4}  '
              f'{counts["another_move"]:>5}  {counts["unknown_wording"]:>7}  '
              f'{counts["degenerate"]:>5}  {counts["distinct"]:>8}')
    total = report['move'] / len(sample)
    print(f'\nmove accuracy {report["move"]}/{len(sample)} = {total:.3f}; '
          f'cell {report["exact"]}/{len(sample)}; copies {report["copied"]}; '
          f'degenerate {report["degenerate"]}; leaked {report["leaked"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
