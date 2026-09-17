"""Build the three-axis Crownless corpus: move x channel x stance.

Every row carries a stance, including the conversational moves that used to be
answered from the bare account. That is the direct fix for voice conditioning
nothing: a villager who cannot see whose mouth the line comes out of has no way
to sound like anyone.

Every row also carries its own `accepted` list -- each wording that would be
correct for its cell. Scoring reads that list rather than comparing against the
one string the generator happened to pick, so adding variety cannot be misread
as regression by a checkpoint gate.
"""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_crownless_v2 import accepted_forms
from crownless_conversation import (VOICE_LINES, MEMORY_LINES, THOUGHT, QUESTIONS,
                                    family, fill, STRESS_PREFIX)
from crownless_moves import (MOVES, CUE, VOICES, STRESS, forms,
                             DISPUTE_OPEN, DISPUTE_CLOSE, tier, situation_marks,
                             social_marks, stance_marks)

GOALS = ('secure_livelihood', 'survive_crisis', 'carry_news', 'keep_order')
LEVELS = ('low', 'medium', 'high')
ASK_CERTAIN = ['How sure are you?', 'Are you sure?', 'Can we trust that account?']
ASK_SOURCE = ['Who told you?', 'Where did that come from?', 'Who carried it to you?']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]


def stance_for(row, rng):
    """A stance for every row. Sim-derived rows keep the character the
    simulation gave them; account rows draw one.

    Drawn, not cycled: an index-derived cycle advances in step with the move
    index and the two share factors, so whole (move, voice) cells never
    co-occur and the phrasings in them are never trained. The draw is seeded,
    so the corpus stays reproducible."""
    mind = row.get('mind')
    if mind and mind.get('goal'):
        voice = row.get('voice')
        return {'goal': mind['goal'], 'stress': mind.get('stress', 'medium'),
                'courage': mind.get('courage', 'medium'),
                'voice': voice if voice in VOICES else 'resident',
                'memories': list(mind.get('memories', [])), 'thoughts': list(mind.get('thoughts', []))}
    return {'goal': rng.choice(GOALS), 'stress': rng.choice(LEVELS),
            'courage': rng.choice(LEVELS), 'voice': rng.choice(VOICES),
            'memories': [], 'thoughts': []}


def move_row(base, rule, move, stance, rng, replacement=None, paraphrase=False):
    row = copy.deepcopy(base)
    own = row['output']
    heard = rng.choice(sorted(accepted_forms(row, rule)))
    voice = stance['voice']
    fam = family(row['kind'])
    pool_voice = voice if voice in VOICE_LINES else 'resident'
    history, prefix, suffix, use_claim, changed = [], '', '', False, None
    accepted = None

    if move == 'open':
        target, use_claim = own, True
    elif move == 'answer':
        history = [{'speaker': 'other', 'text': 'Tell me the news.' if paraphrase else rng.choice(QUESTIONS)}]
        target, use_claim = own, True
    elif move == 'remark':
        history = [{'speaker': 'other', 'text': heard}]
        accepted = list(VOICE_LINES.get(pool_voice, VOICE_LINES['resident']).get(
            fam, VOICE_LINES.get(pool_voice, VOICE_LINES['resident'])['neutral']))
        accepted = [fill(x, row) for x in accepted]
        target = rng.choice(accepted)
    elif move == 'affirm':
        history = [{'speaker': 'other', 'text': heard}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'dispute':
        candidates = [f for f in row['fields'] if f.get('spoken') and f['knowledge'] != 3 and
                      f['text'] in heard and f['role'] not in (0, 8)]
        if candidates:
            field = rng.choice(candidates)
            alternative = replacement(field) if replacement else 'Farhaven'
            if alternative == field['text']: alternative = 'Another place'
            heard = heard.replace(field['text'], alternative)
            changed = {'field': field['field'], 'own': field['text'], 'heard': alternative}
        history = [{'speaker': 'other', 'text': heard}]
        prefix = rng.choice(DISPUTE_OPEN[stance['stress']])
        suffix = rng.choice(DISPUTE_CLOSE[stance['stress']])
        target, use_claim = prefix + own + suffix, True
        accepted = [o + own + c for o in DISPUTE_OPEN[stance['stress']]
                    for c in DISPUTE_CLOSE[stance['stress']]]
    elif move == 'hedge':
        history = [{'speaker': 'self', 'text': own},
                   {'speaker': 'other', 'text': 'How certain is that?' if paraphrase else rng.choice(ASK_CERTAIN)}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'attribute':
        history = [{'speaker': 'self', 'text': own},
                   {'speaker': 'other', 'text': 'Where did that story come from?' if paraphrase else rng.choice(ASK_SOURCE)}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'defer':
        history = [{'speaker': 'self', 'text': rng.choice(ASK_CERTAIN)},
                   {'speaker': 'other', 'text': rng.choice(forms('hedge', row, stance))}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'settle':
        history = [{'speaker': 'self', 'text': rng.choice(forms('defer', row, stance))},
                   {'speaker': 'other', 'text': rng.choice(forms('hedge', row, stance))}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'part':
        history = [{'speaker': 'self', 'text': rng.choice(forms('settle', row, stance))},
                   {'speaker': 'other', 'text': rng.choice(forms('settle', row, stance))}]
        accepted = forms(move, row, stance); target = rng.choice(accepted)
    elif move == 'recall':
        history = [{'speaker': 'other', 'text': heard}]
        lines = [fill(x, row) for x in MEMORY_LINES.get(fam, MEMORY_LINES['neutral'])]
        line = rng.choice(lines)
        if stance['memories']:
            memory = rng.choice(stance['memories'])
            if len(memory) > 80: memory = memory[:80].rsplit(' ', 1)[0] + '...'
            connectors = [' It reminds me of when ', ' It puts me in mind of ',
                          ' Like when ', ' It brings back ']
            target = line + rng.choice(connectors) + memory
            accepted = [l + c + memory for l in lines for c in connectors]
        else:
            target, accepted = line, lines
    elif move == 'muse':
        pool = THOUGHT.get(stance['goal'], {}).get(fam, ['I should keep an eye on how this turns out.'])
        pool = [fill(x, row) for x in pool]
        heads = STRESS_PREFIX[stance['stress']]
        accepted = [h + t for h in heads for t in pool]
        target = rng.choice(accepted)
    else:
        raise ValueError(move)

    # Openings stack in layers: a predicament, then company, then mettle and
    # purpose. Each layer is [''] unless its marked state holds, so an
    # unmarked move composes over the empty string and comes out byte-identical
    # down to the rng stream.
    layers = [situation_marks(move, stance.get('situation')),
              social_marks(move, stance.get('social')),
              stance_marks(move, stance)]
    if any(layer != [''] for layer in layers):
        combined = ['']
        for layer in layers:
            combined = [p + q for p in combined for q in layer]
        if move == 'open':
            # The prefix variable feeds the copy-span offset below, so the
            # chosen mark goes through it rather than around it.
            prefix = rng.choice(combined)
            target, accepted = prefix + own, [m + own for m in combined]
        else:
            accepted = [m + t for m in combined for t in accepted]
            target = rng.choice(accepted)

    # Conversations reach the model several turns deep and the encoder keeps
    # the last four.
    for _ in range(rng.choice((0, 1, 1, 2)) if history else 0):
        first = history[0]['speaker']
        second = 'self' if first == 'other' else 'other'
        history = [{'speaker': first, 'text': rng.choice(QUESTIONS)},
                   {'speaker': second, 'text': own if second == 'self' else heard}] + history

    channel = MOVES[move]
    row.update(output=target, history=history, act=move, move=move, channel=channel,
               control=CUE[move], changed=changed,
               voice=voice, accepted=sorted(set(accepted or [target])),
               mind={'goal': stance['goal'], 'stress': stance['stress'], 'courage': stance['courage'],
                     'memories': list(stance['memories']), 'thoughts': list(stance['thoughts'])})
    if use_claim:
        # Where the account actually sits in the finished target, rather than
        # the length of a variable that now holds only the dispute opener: the
        # stacked situation/company/mettle marks prepend text too, and a claim
        # move may carry any of them. The account appears verbatim once, since
        # no opener or closer pool contains it.
        offset = len(target[:target.index(own)].encode())
        for span in row['copies']:
            span['start'] += offset
            span['end'] += offset
    else:
        row['copies'] = []
    return row


def build_split(bases, rules, seed, repeats=1, paraphrase=False, holdout=(),
                confusable=False, situation=False, social=False):
    rng = random.Random(seed)
    pools = {}
    for row in bases:
        for f in row['fields']:
            if f.get('spoken') and f['knowledge'] != 3: pools.setdefault(f['role'], set()).add(f['text'])
    pools = {role: sorted(values) for role, values in pools.items()}
    def replacement(field):
        options = [x for x in pools[field['role']] if x != field['text']]
        if not options: return 'A different account'
        if confusable:
            # Near-miss swaps: the telling differs by a letter or two, so the
            # row is solvable only by reading the span, not the shape of it.
            options = sorted(options,
                             key=lambda x: difflib.SequenceMatcher(None, x, field['text']).ratio(),
                             reverse=True)[:3]
        return rng.choice(options)
    order = list(MOVES)
    result = []
    for repetition in range(repeats):
        for i, base in enumerate(bases):
            move = order[(i // 2 + repetition * 5) % len(order)]
            stance = stance_for(base, rng)
            # Held-out cells stay out of every repetition, or the holdout is
            # a lie told to the eval. The skip consumes the stance draw above,
            # so a held-out build is reproducible but not prefix-identical to
            # a full one; the manifest records the set either way.
            if (move, stance['voice'], stance['stress']) in holdout: continue
            if situation:
                # Drawn here, beside the stance, so move_row sees the same dict
                # the row keeps. Fabricated like the BALANCE stance overrides,
                # for the same reason: uniform coverage beats sim-faithful
                # rarity when the point is teaching the conditioning, and the
                # seed pins the draws.
                stance['situation'] = {'hungry': rng.random() < 0.5,
                                       'sheltered': rng.random() < 0.5,
                                       'in_transit': rng.random() < 0.5}
            if social:
                stance['social'] = {
                    'owes_listener': rng.random() < 0.5,
                    'trusts_listener': rng.random() < 0.5,
                    'faction': rng.choice([None, 'crown', 'guild', 'commons']),
                    'far_from_home': rng.random() < 0.5}
            row = move_row(base, rules[base['rule']], move, stance, rng, replacement, paraphrase)
            row['id'] = f"{base['id']}:{move}:{repetition}"
            if situation:
                row['situation'] = stance['situation']
            if social:
                row['social'] = stance['social']
            result.append(row)
    return result


def report(rows):
    moves = Counter(r['move'] for r in rows)
    variety = {m: len({r['output'] for r in rows if r['move'] == m}) for m in sorted(moves)}
    return {'rows': len(rows), 'moves': dict(sorted(moves.items())), 'distinct_outputs': variety,
            'channels': dict(Counter(r['channel'] for r in rows)),
            'voices': len({r['voice'] for r in rows}), 'kinds': len({r['kind'] for r in rows}),
            'with_stance': sum(1 for r in rows if r.get('mind')),
            'history_depth': dict(sorted(Counter(len(r['history']) for r in rows).items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accounts', type=Path, required=True)
    p.add_argument('--mind', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--holdout-cells', default='',
                   help="Comma-separated move:voice:stress cells kept out of train, "
                        "so the eval can tell memory from generalization")
    p.add_argument('--confusable-replacements', action='store_true',
                   help='Swap near-miss field values instead of arbitrary ones')
    p.add_argument('--situation', action='store_true',
                   help='Give every row a hungry/sheltered/in_transit situation, drawn '
                        'uniformly. Fabricated like the stance overrides so coverage '
                        'is uniform; the seed pins the draws.')
    p.add_argument('--social', action='store_true',
                   help='Give every row an owes/trusts/faction/far situation, drawn '
                        'uniformly (faction uniform over absent/crown/guild/commons). '
                        'Same fabrication rationale as --situation.')
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
    holdout = {tuple(cell.split(':')) for cell in args.holdout_cells.split(',') if cell}
    if any(len(cell) != 3 for cell in holdout): p.error('Holdout cells read move:voice:stress')
    for split in ('train', 'validation', 'test'):
        repeats = args.repeats if split == 'train' else 1
        kept = holdout if split == 'train' else ()
        splits[split] = (build_split(accounts[split], rules, f'{args.seed}:{split}:account', repeats,
                                     holdout=kept, confusable=args.confusable_replacements,
                                     situation=args.situation, social=args.social) +
                         build_split(minds[split], rules, f'{args.seed}:{split}:mind', repeats,
                                     holdout=kept, confusable=args.confusable_replacements,
                                     situation=args.situation, social=args.social))
    splits['wording'] = (build_split(accounts['test'], rules, f'{args.seed}:wording:account', paraphrase=True,
                                     situation=args.situation, social=args.social) +
                         build_split(minds['test'], rules, f'{args.seed}:wording:mind', paraphrase=True,
                                     situation=args.situation, social=args.social))

    files = {}
    for split, rows in splits.items():
        path = args.output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        files[split] = {'sha256': sha(path), **report(rows)}
    manifest = {
        'schema': 'crownless.core_moves.v1', 'seed': args.seed, 'repeats': args.repeats,
        'axes': {'move': list(MOVES), 'channel': sorted(set(MOVES.values())),
                 'stance': ['voice', 'goal', 'stress', 'courage'],
                 'situation': ['hungry', 'sheltered', 'in_transit'],
                 'social': ['owes_listener', 'trusts_listener', 'faction', 'far_from_home']},
        'voices': list(VOICES), 'stress': list(STRESS),
        'accounts_manifest_sha256': sha(args.accounts / 'manifest.json'),
        'mind_manifest_sha256': sha(args.mind / 'manifest.json'),
        'rules_sha256': sha(args.accounts / 'rules.json'),
        'holdout_cells': sorted(':'.join(cell) for cell in holdout),
        'confusable_replacements': args.confusable_replacements,
        'situation': args.situation,
        'social': args.social,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_hashes': {n: sha(Path(__file__).with_name(n)) for n in
                          ('crownless_moves.py', 'build_crownless_moves.py', 'crownless_v2.py')},
        'splits': files,
        'scope': 'Move x channel x stance over shared account rows. Every row carries a stance '
                 'and the full set of wordings accepted for its cell.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    t = files['train']
    print(json.dumps({'train_rows': t['rows'], 'moves': len(t['moves']),
                      'distinct_outputs': t['distinct_outputs']}, indent=2))


if __name__ == '__main__': main()
