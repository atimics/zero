"""Amplify one stance line at inference with classifier-free guidance.

The shipped model reads `# stress:` in 11% of prompts and `# voice:` in 46%.
Classifier-free guidance (Ho and Salimans 2022; Sanchez et al. 2024 for pure
language models) turns that into a knob rather than a retrain: run the model
once on the prompt as written and once on the same prompt with the line set to
its other values, then extrapolate away from the average of the others.

The contrast is the other values of the same line rather than a prompt with the
line removed. This model was trained with the stance always present, so a
stance-free prompt is out of distribution for it and its logits are not a
trustworthy baseline to extrapolate from.

The copy decision is taken from the conditional stream alone. Guidance is for
choosing wordings; the account's spans are already carried correctly and there
is nothing to gain by perturbing them.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row, typed_stance_for
from crownless_v2_export import load_export

AXES = {'stress': ('low', 'medium', 'high'),
        'courage': ('low', 'medium', 'high'),
        'goal': ('secure_livelihood', 'survive_crisis', 'carry_news', 'keep_order'),
        'voice': ('baker', 'scribe', 'farmer', 'smith', 'innkeeper', 'miller', 'shepherd',
                  'woodcutter', 'resident')}


def variants(row, axis):
    """The row as written, followed by the same row at the axis's other values."""
    if axis == 'voice':
        own = row.get('voice')
        return [row] + [{**row, 'voice': v} for v in AXES[axis] if v != own]
    own = row['mind'].get(axis)
    return [row] + [{**row, 'mind': {**row['mind'], axis: v}} for v in AXES[axis] if v != own]


def guided(model, tokenizer, records, weight, device='cpu', max_tokens=160):
    model.eval()
    streams = []
    for record in records:
        inputs = batch([record], device)
        n = record['prefix_length']
        hidden, caches = model.hidden(inputs['tokens'][:, :n], inputs['meta'][:, :n])
        streams.append({'inputs': inputs, 'source': hidden, 'current': hidden[:, -1:],
                        'caches': caches, 'used': n})
    lead, record = streams[0], records[0]
    generated, stopped = [], False
    blocked = [tokenizer.token_to_id(f'[F{i}]') for i in range(8)]
    end = tokenizer.token_to_id('[EOS]')

    for _ in range(max_tokens):
        heads = [model.heads(s['current'], s['inputs']['candidates'],
                             s['inputs']['candidate_mask'], s['source']) for s in streams]
        logits, gate, scores = heads[0]
        if record['source_ids'] and gate.item() > 0:
            choice = scores[0, -1].argmax().item()
            tokens = record['source_ids'][choice]
            feedback = record['feedback_ids'][choice]
        else:
            conditional = logits[0, -1]
            if len(heads) > 1 and weight != 1.0:
                contrast = torch.stack([h[0][0, -1] for h in heads[1:]]).mean(0)
                conditional = contrast + weight * (conditional - contrast)
            for marker in blocked:
                if marker is not None: conditional[marker] = -torch.inf
            token = conditional.argmax().item()
            if token == end:
                stopped = True
                break
            tokens = feedback = [token]
        if lead['used'] + len(feedback) > model.config.context: break
        generated.extend(tokens)
        step = torch.tensor([feedback], device=device)
        meta = torch.zeros(1, len(feedback), 5, dtype=torch.long, device=device)
        for s in streams:
            s['current'], s['caches'] = model.hidden(step, meta, s['caches'])
            s['current'] = s['current'][:, -1:]
            s['used'] += len(feedback)
    return {'text': tokenizer.decode(generated), 'stopped': stopped}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--axis', choices=sorted(AXES), default='stress')
    p.add_argument('--weights', type=float, nargs='+', default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument('--rows-per-move', type=int, default=10)
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    model, meta = load_export(args.model, args.tokenizer, 'cpu')
    model.mode = 'conversation'
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(line) for line in (args.corpus / 'validation.jsonl').read_text().splitlines()]
    rows = [{**r, 'kind_id': meta['meaning_ids'][r['rule']]} for r in rows]

    taken = {}
    for row in rows:
        seen = taken.setdefault(row['move'], [])
        if len(seen) < args.rows_per_move: seen.append(row)
    sample = [row for move in sorted(taken) for row in taken[move]]

    # Every wording the corpus writes for a move, and for that move at this row's cell.
    shapes = {}
    for row in rows:
        shapes.setdefault(row['move'], set()).update(row.get('accepted', [row['output']]))

    report = {}
    for weight in args.weights:
        move_hits = cell_hits = stopped = 0
        for row in sample:
            records = [encode_row(tokenizer, v, slots=True, conversation=True,
                                 typed_stance=typed_stance_for(model))
                       for v in variants(row, args.axis)]
            out = guided(model, tokenizer, records, weight)
            stopped += out['stopped']
            move_hits += out['text'] in shapes[row['move']]
            cell_hits += out['text'] in row.get('accepted', [row['output']])
        n = len(sample)
        report[weight] = {'move': move_hits, 'cell': cell_hits, 'stopped': stopped, 'rows': n}
        print(f'guidance {weight:>4}  move {move_hits:>4}/{n}  cell {cell_hits:>4}/{n}  '
              f'stopped {stopped}/{n}', flush=True)
    if args.output: args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
