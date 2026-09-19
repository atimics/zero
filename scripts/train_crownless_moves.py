"""Tune the 5M core on the three-axis corpus: move x channel x stance.

Scoring reads each row's own `accepted` pool rather than the single string the
generator picked. That change has to come first: with several correct wordings
per cell, exact match against one of them falls as variety rises, and a gate
reading it would reject precisely the model this corpus exists to build.

The guard is no longer relative to the shipped model. Every row now carries a
stance, so the prompt shape itself changed and the old model is out of
distribution on it -- its exact score says nothing. What survives the change is
what actually broke before: collapse, leaked scaffolding, early stopping, and
now a variety floor, since a model that answers every stance with one sentence
is the failure this whole redesign is aimed at.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row, generate
from crownless_v2_export import load_export, export
from crownless_moves import MOVES

GOALS = ('secure_livelihood', 'survive_crisis', 'carry_news', 'keep_order')
LEVELS = ('low', 'medium', 'high')
VOICES = ('baker', 'scribe', 'farmer', 'smith', 'shepherd', 'miller', 'resident')
REPEAT = re.compile(r'\b(\w+)(?:\s+\1\b){2,}')
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_commit():
    """A checkout answers for itself; a packaged run is told what it was built
    from. A cloud job has neither a repository nor a remote to ask."""
    named = os.environ.get('CROWNLESS_SOURCE_COMMIT')
    if named: return named
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except (subprocess.SubprocessError, OSError):
        return 'unknown'


def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def encoded(tokenizer, row, typed=False, situ=False, social=False):
    return encode_row(tokenizer, row, slots=True, conversation=True, typed_stance=typed,
                      situation=situ, social=social)


def attach_stance(model):
    """Grow the four stance tables onto a checkpoint that was trained without
    them. The base performs the moves already; what it cannot do is read the
    stance, so the tables start fresh and everything else carries over."""
    dim = model.config.dim
    model.config = replace(model.config, voices=16, goals=8, levels=4)
    model.voices = nn.Embedding(16, dim)
    model.goals = nn.Embedding(8, dim)
    model.stresses = nn.Embedding(4, dim)
    model.courages = nn.Embedding(4, dim)
    for table in (model.voices, model.goals, model.stresses, model.courages):
        nn.init.normal_(table.weight, std=.02)
        table.to(next(model.parameters()).device)
    return model


def attach_situation(model):
    """Grow the three binary situation tables onto a typed checkpoint. Same
    deal as stance: fresh tables, everything else carries over."""
    dim = model.config.dim
    model.config = replace(model.config, situations=2)
    model.hungry = nn.Embedding(2, dim)
    model.sheltered = nn.Embedding(2, dim)
    model.in_transit = nn.Embedding(2, dim)
    for table in (model.hungry, model.sheltered, model.in_transit):
        nn.init.normal_(table.weight, std=.02)
        table.to(next(model.parameters()).device)
    return model


def attach_social(model):
    """Grow the four company tables onto a checkpoint that reads stance and
    situation but knows nothing of debts, trust, faction or distance. The
    faction table has four rows (absent/crown/guild/commons); the rest are
    binary. Fresh tables, everything else carries over."""
    dim = model.config.dim
    model.config = replace(model.config, socials=4)
    model.owes = nn.Embedding(2, dim)
    model.trusts = nn.Embedding(2, dim)
    model.faction = nn.Embedding(4, dim)
    model.far = nn.Embedding(2, dim)
    for table in (model.owes, model.trusts, model.faction, model.far):
        nn.init.normal_(table.weight, std=.02)
        table.to(next(model.parameters()).device)
    return model


def vocabulary(*groups):
    """Every word the corpus writes plus the words of the prompts it writes
    from. Anything outside it is the model inventing spelling."""
    known = set()
    for rows in groups:
        for row in rows:
            for field in ('output', 'prefix'):
                known.update(w.lower() for w in WORD.findall(row.get(field, '')))
            for span in row.get('fields', []) + row.get('copies', []):
                known.update(w.lower() for w in WORD.findall(span.get('text', '')))
            for line in row.get('history', []):
                known.update(w.lower() for w in WORD.findall(line.get('text', '')))
    return known


def bridged(row, rng):
    """A plain account put under a mind context. Mind acts only cover the 21
    kinds the simulation gossips about; the conversation half covers all 44, so
    wrapping those rows teaches that the context does not change what is known."""
    return {**row, 'voice': rng.choice(VOICES), 'control': 'say', 'act': 'bridge',
            'mind': {'goal': rng.choice(GOALS), 'stress': rng.choice(LEVELS),
                     'courage': rng.choice(LEVELS), 'memories': [], 'thoughts': []}}


def shape(text, row):
    """A wording with the copied spans blanked out. Two rows of the same move
    differ by the names they carry; their shape is what the move sounds like."""
    for span in row.get('copies', []):
        text = text.replace(span['text'], '<>')
    return text


def shapes_by_move(*groups):
    """Every shape each move is written in, across the whole corpus. A reply
    that matches one of these performed the move; whether it also matched the
    stance and voice of this particular row is the separate, harder question
    that `exact` asks."""
    known = {}
    for rows in groups:
        for row in rows:
            if 'move' not in row: continue
            known.setdefault(row['move'], set()).update(
                shape(wording, row) for wording in row.get('accepted', [row['output']]))
    return known


def quality(row, text, stopped, rules, known=None, moves=None):
    """Exact match cannot tell a fair paraphrase from a collapse, and repetition
    is only one shape of collapse: 'It wellerve bea fim' carries no repeat, so
    the word list is what catches it.

    `exact` asks for this row's cell: the wording this move takes at this
    stance, in this voice, out of as many as 568 the move is written in. A
    model that has learned the move and missed the cell scores as low there as
    one emitting nonsense -- 3.7% on affirm, 0.8% on open -- so `move` is
    reported beside it, and the collapse checks below are what separate the two
    failures."""
    words = text.split()
    local = (vocabulary([row]) | known) if known else None
    invented = [w for w in WORD.findall(text) if w.lower() not in local] if local else []
    copies = [span['text'] for span in row.get('copies', [])]
    return {'exact': stopped and text in row.get('accepted', [row['output']]),
            'move': bool(stopped and moves and shape(text, row) in moves.get(row.get('move'), ())),
            'stopped': stopped, 'leaked': '# ' in text,
            'degenerate': bool(REPEAT.search(text)) or len(invented) >= 2 or
                          (len(words) > 3 and len(set(words)) < len(words) / 2),
            'invented': invented[:4], 'empty': not text.strip(),
            'copied': all(span in text for span in copies) if copies else None}


def evaluate(model, tokenizer, rows, rules, label, known=None, moves=None, typed=False,
             situ=False, social=False):
    """Runs on an already-CPU model. Moving the training model across devices
    mid-run strands the optimiser state on the accelerator and thrashes it."""
    model.eval()
    scored = []
    for row in rows:
        result = generate(model, tokenizer, encoded(tokenizer, row, typed, situ, social))
        scored.append({'id': row['id'], 'act': row.get('act', 'account'),
                       'reference': row['output'], 'text': result['text'],
                       **quality(row, result['text'], result['stopped'], rules, known, moves)})
    copied = [x['copied'] for x in scored if x['copied'] is not None]
    spread = {a: len({x['text'] for x in scored if x['act'] == a})
              for a in sorted({x['act'] for x in scored})}
    return {'label': label, 'count': len(scored), 'distinct': spread,
            'thinnest': min(spread.values()) if spread else 0,
            'exact': sum(x['exact'] for x in scored),
            'move': sum(x['move'] for x in scored),
            'stopped': sum(x['stopped'] for x in scored),
            'leaked': sum(x['leaked'] for x in scored),
            'degenerate': sum(x['degenerate'] for x in scored),
            'empty': sum(x['empty'] for x in scored),
            'copied': f'{sum(copied)}/{len(copied)}' if copied else 'n/a',
            'copied_all': all(copied) if copied else True,
            'acts': {act: {'count': sum(x['act'] == act for x in scored),
                           'exact': sum(x['act'] == act and x['exact'] for x in scored),
                           'move': sum(x['act'] == act and x['move'] for x in scored),
                           'degenerate': sum(x['act'] == act and x['degenerate'] for x in scored)}
                     for act in sorted({x['act'] for x in scored})}}, scored


def loss_over(model, records, device, size=16):
    with torch.no_grad():
        values = [model.loss(batch(records[i:i + size], device)).item()
                  for i in range(0, len(records), size)]
    return sum(values) / len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True, help='Thirteen-act corpus directory')
    p.add_argument('--chat', type=Path, help='Recorded multi-turn rows the guard protects')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--bridge-ratio', type=float, default=0.10)
    p.add_argument('--distill', type=float, default=2.0)
    p.add_argument('--typed-stance', action='store_true',
                   help='Carry voice, goal, stress and courage as typed meta ids rather than '
                        'four lines of prompt text. Grows four embedding tables onto the base.')
    p.add_argument('--situation', action='store_true',
                   help='Carry hungry, sheltered and in_transit as three binary meta ids, '
                        'always present. Grows three embedding tables onto a typed base.')
    p.add_argument('--social', action='store_true',
                   help='Carry owes_listener, trusts_listener, faction and far_from_home: '
                        'three binaries always present plus a categorical faction id '
                        '(0 absent, gated like voice). Grows four tables onto a '
                        'situation-reading base.')
    p.add_argument('--anchor', choices=('bridge', 'spoken'), default='bridge',
                   help='What the distillation holds the student to: the plain account the base '
                        'answers well, or the spoken-move rows it has never answered at all')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--guard-rows', type=int, default=48)
    p.add_argument('--guard-every', type=int, default=500)
    p.add_argument('--accept-move-rate', type=float, default=0.83,
                   help='Share of guard rows whose reply must be a known wording of the '
                        'row\'s move. The gate proper: the base scores 9/48 here.')
    p.add_argument('--accept-rate', type=float, default=0.0,
                   help='Share of guard rows whose reply must be the wording this row\'s '
                        'stance and voice call for, out of as many as 568 the move is '
                        'written in. Reported rather than gated until a run measures what '
                        'is reachable; raise it once one has.')
    p.add_argument('--variety', type=int, default=2,
                   help='Distinct replies the thinnest move must reach on the guard set')
    p.add_argument('--eval-rows', type=int, default=260)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed)

    model, metadata = load_export(args.base, args.tokenizer, args.device)
    model.mode = 'conversation'
    if args.typed_stance and model.voices is None: attach_stance(model)
    if args.situation and getattr(model, 'hungry', None) is None: attach_situation(model)
    if args.social and getattr(model, 'owes', None) is None: attach_social(model)
    assert sum(x.numel() for x in model.parameters()) <= 5000000
    teacher, _ = load_export(args.base, args.tokenizer, args.device)
    teacher.mode = 'conversation'; teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.corpus / 'rules.json').read_text())['rules']}
    if sha(args.corpus / 'rules.json') != metadata['rules_sha256']: p.error('Grammar differs')
    meanings = metadata['meaning_ids']

    corpus = {}
    for split in ('train', 'validation', 'test', 'wording'):
        corpus[split] = [{**row, 'kind_id': meanings[row['rule']]}
                         for row in read(args.corpus / f'{split}.jsonl')]
    chat = [{**row, 'kind_id': meanings[row['rule']]}
            for row in read(args.chat)] if args.chat else []
    # What the shipped model answers well is the plain account, not the moves:
    # it was never trained on nine of the twelve, and measured on a balanced
    # sample it performs exactly the four it already had. Anchoring on
    # spoken-move rows therefore asks a teacher about moves it has never made,
    # over every output token of every reply -- 41,996 of 41,996 measured -- and
    # the four it knows are the four that score. `bridge` anchors on the plain
    # account under a mind context instead: the same rows, stripped to the
    # behaviour the anchor exists to protect. `spoken` reproduces the runs that
    # plateaued at 19/48.
    talk = [r for r in corpus['train'] if MOVES[r['move']] == 'spoken']
    known = vocabulary(*corpus.values(), chat)
    move_shapes = shapes_by_move(*corpus.values(), chat)

    args.output.mkdir(parents=True)
    (args.output / 'tokenizer.json').write_bytes(args.tokenizer.read_bytes())
    manifest = {'source_commit': source_commit(),
                'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                                  ('crownless_v2.py', 'crownless_conversation.py', 'train_crownless_moves.py')},
                'base_sha256': sha(args.base), 'tokenizer_sha256': sha(args.tokenizer),
                'corpus_manifest_sha256': sha(args.corpus / 'manifest.json'),
                'rows': {'train': len(corpus['train']), 'conversation': len(talk)},
                'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
                'bridge_ratio': args.bridge_ratio, 'distill': args.distill, 'lr': args.lr,
                'anchor': args.anchor, 'typed_stance': args.typed_stance,
                'situation': args.situation, 'social': args.social,
                'moves': list(MOVES), 'variety_floor': args.variety,
                'contract': {'move_rate': args.accept_move_rate, 'cell_rate': args.accept_rate,
                             'variety': args.variety, 'guard_rows': args.guard_rows,
                             'hard': ['no degenerate', 'no leaked', 'all stopped',
                                      'every copy-bearing row keeps its spans'],
                             'declared': 'Move accuracy gates; cell accuracy is recorded for '
                                         'the next contract to set. Thresholds are fixed '
                                         'before the run and not moved during it.'},
                'scope': 'Thirteen acts in one corpus, anchored to the plain account the '
                         'shipped model answers well, and gated on generated answers rather '
                         'than loss.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    # The corpus is written conversation-half first, so a head slice would score
    # only one act family. Sample across the whole split instead.
    sample = random.Random(args.seed).sample(corpus['validation'], 244)
    validation = [encoded(tokenizer, r, args.typed_stance, args.situation, args.social) for r in sample]
    talk_validation = [encoded(tokenizer, r, args.typed_stance, args.situation, args.social) for r in
                       [x for x in corpus['validation'] if MOVES[x['move']] == 'spoken'][:244]]
    # Taking the head of the spoken rows left recall and muse unscored -- the two
    # moves the shipped model already answers -- and let the row order decide the
    # balance. Round-robin by move instead, so every move is represented equally.
    by_move = {}
    for row in corpus['validation']:
        by_move.setdefault(row['move'], []).append(row)
    guard = [row for group in zip(*(by_move[m] for m in sorted(by_move)))
             for row in group][:args.guard_rows]

    scout, _ = load_export(args.base, args.tokenizer, 'cpu')
    scout.mode = 'conversation'
    if args.typed_stance and scout.voices is None: attach_stance(scout)
    if args.situation and getattr(scout, 'hungry', None) is None: attach_situation(scout)
    if args.social and getattr(scout, 'owes', None) is None: attach_social(scout)
    base_report, _ = evaluate(scout, tokenizer, guard, rules, 'base', known, move_shapes, args.typed_stance, args.situation, args.social)
    floor = max(0, round(len(guard) * args.accept_rate))
    move_floor = max(0, round(len(guard) * args.accept_move_rate))
    chat_report, _ = evaluate(scout, tokenizer, chat, rules, 'chat', known, move_shapes, args.typed_stance, args.situation, args.social) if chat else (None, None)
    print(json.dumps({'base_guard': base_report, 'exact_floor': floor,
                      'base_chat': chat_report,
                      'base_validation': loss_over(teacher, validation, args.device)}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = random.Random(args.seed)
    bridge_count = max(1, round(args.batch_size * args.bridge_ratio))
    best, best_step, history = float('inf'), 0, []
    candidate = float('inf')
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        rows = rng.choices(corpus['train'], k=args.batch_size - bridge_count)
        rows += [bridged(r, rng) for r in rng.choices(talk, k=bridge_count)]
        anchor_rows = rng.choices(talk, k=max(2, bridge_count))
        if args.anchor == 'bridge':
            anchor_rows = [bridged(row, rng) for row in anchor_rows]
        for group in optimizer.param_groups:
            group['lr'] = args.lr * min(step / 100, 1) * (.1 + .9 * (1 - step / args.steps))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch([encoded(tokenizer, r, args.typed_stance, args.situation, args.social) for r in rows], args.device))
        anchor = batch([encoded(tokenizer, r) for r in anchor_rows], args.device)  # untyped: the teacher cannot read a stance id
        student_hidden, _ = model.hidden(anchor['tokens'], anchor['meta'])
        student_logits, _, _ = model.heads(student_hidden, anchor['candidates'], anchor['candidate_mask'])
        with torch.no_grad():
            teacher_hidden, _ = teacher.hidden(anchor['tokens'], anchor['meta'])
            teacher_logits, _, _ = teacher.heads(teacher_hidden, anchor['candidates'], anchor['candidate_mask'])
        scored = (anchor['copy_labels'] != -100).flatten()
        distill = F.kl_div(F.log_softmax(student_logits.flatten(0, 1)[scored], -1),
                           F.softmax(teacher_logits.flatten(0, 1)[scored], -1), reduction='batchmean')
        total = loss + args.distill * distill
        if not torch.isfinite(total): raise ValueError('Non-finite loss')
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()

        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            val = loss_over(model, validation, args.device)
            talk_val = loss_over(model, talk_validation, args.device)
            item = {'step': step, 'loss': loss.item(), 'distill': distill.item(),
                    'validation': val, 'conversation_validation': talk_val}
            accepted = False
            if val < best and step % args.guard_every == 0:
                scout.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
                report, _ = evaluate(scout, tokenizer, guard, rules, 'guard', known, move_shapes, args.typed_stance, args.situation, args.social)
                accepted = (report['move'] >= move_floor and report['exact'] >= floor
                            and not report['degenerate'] and not report['leaked']
                            and report['stopped'] == report['count'] and report['copied_all']
                            and report['thinnest'] >= args.variety)
                item['guard'] = {k: report[k] for k in
                                 ('move', 'exact', 'copied', 'degenerate', 'leaked', 'thinnest')}
                if chat:
                    turns, _ = evaluate(scout, tokenizer, chat, rules, 'chat', known, move_shapes, args.typed_stance, args.situation, args.social)
                    accepted = (accepted and not turns['degenerate'] and not turns['leaked']
                                and turns['stopped'] == turns['count'])
                    item['chat'] = {k: turns[k] for k in ('exact', 'stopped', 'degenerate', 'leaked')}
            item['accepted'] = accepted
            item['seconds'] = time.monotonic() - start
            history.append(item); print(json.dumps(item), flush=True)
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            if args.device == 'mps': torch.mps.empty_cache()
            # A run that never clears the gate costs the same hours as one that
            # does, and its weights are what the next decision reads. Keep the
            # lowest-validation checkpoint whichever way the gate went; best.pt
            # stays the gated one, so nothing downstream can confuse the two.
            if val < candidate:
                candidate = val
                torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'step': step, 'validation': val, 'accepted': accepted,
                            'guard': item.get('guard'), 'chat': item.get('chat')},
                           args.output / 'candidate.pt')
            if accepted:
                best, best_step = val, step
                torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'step': step}, args.output / 'best.pt')
    if best_step == 0:
        # Export the candidate too: the .ccv2 is what the scorers read, and what
        # the native runtime reads once compile_model.py re-pins the file hash.
        # The name keeps a run that failed its gate off the shipping path.
        if (args.output / 'candidate.pt').exists():
            held = torch.load(args.output / 'candidate.pt', map_location='cpu', weights_only=True)
            model.to('cpu').load_state_dict(held['state'])
            export(model, args.tokenizer, args.output / 'rejected.ccv2',
                   {k: metadata[k] for k in ['meaning_ids', 'kind_ids', 'rules_sha256']} |
                   {'step': held['step'], 'seed': args.seed})
            report = {'rejected': True, 'base_guard': base_report, 'reason': 'No checkpoint cleared the declared gate: move accuracy, copies, collapse and variety',
                      'candidate_step': held['step'], 'validation': held['validation'],
                      'guard': held.get('guard'), 'chat': held.get('chat'),
                      'model_sha256': sha(args.output / 'rejected.ccv2')}
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        raise SystemExit('No checkpoint cleared the declared gate: move accuracy, copies, collapse and variety')

    saved = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.to('cpu').load_state_dict(saved['state'])
    export(model, args.tokenizer, args.output / 'core.ccv2',
           {k: metadata[k] for k in ['meaning_ids', 'kind_ids', 'rules_sha256']} |
           {'step': saved['step'], 'seed': args.seed})
    compact, _ = load_export(args.output / 'core.ccv2', args.tokenizer)

    bridge_rng = random.Random(args.seed + 1)
    reports = {}
    for label, rows in (('test', corpus['test'][:args.eval_rows]),
                        ('wording', corpus['wording'][:args.eval_rows]),
                        ('bridge', [bridged(r, bridge_rng) for r in
                                    [x for x in corpus['test'] if MOVES[x['move']] == 'spoken'][:args.eval_rows]]),
                        ('chat', chat)):
        if not rows: continue
        report, scored = evaluate(compact, tokenizer, rows, rules, label, known, move_shapes, args.typed_stance, args.situation, args.social)
        (args.output / f'{label}-results.json').write_text(
            json.dumps({'report': report, 'rows': scored}, indent=2) + '\n')
        reports[label] = report
        print(json.dumps(report), flush=True)
    summary = {'best_step': saved['step'], 'model_sha256': sha(args.output / 'core.ccv2'),
               'base_guard': base_report, 'reports': reports}
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'model_sha256': summary['model_sha256'], 'best_step': saved['step']}), flush=True)


if __name__ == '__main__': main()
