"""Tune the 5M core on mind context without losing the plain account voice.

The first mind run trained on 50k mind-only rows at 2e-4 and scored every
checkpoint on mind validation alone. It reached 0.05 validation and forgot how
to answer a bare account: the shape it never saw during tuning collapsed into
repetition. This run keeps the old task present three ways -- replayed rows in
every batch, a frozen teacher anchoring the logits on those rows, and a
checkpoint gate that refuses any step which makes the plain account worse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row, generate
from crownless_v2_export import load_export, export
from crownless_conversation import build_rows, approved_response


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def encoded(tokenizer, row): return encode_row(tokenizer, row, slots=True, conversation=True)


def legacy_rows(path, meanings):
    """Plain account rows, kept in the shape CcCoreModelBegin emits: no mind
    lines, no control cue. The base model is in distribution here, which is
    what makes it usable as a distillation teacher."""
    rows = []
    for row in read(path):
        if row['rule'] not in meanings: continue
        rows.append({**row, 'kind_id': meanings[row['rule']], 'history': [], 'act': 'account'})
    return rows


REPEAT = re.compile(r'\b(\w+)(?:\s+\1\b){2,}')
GOALS = ('secure_livelihood', 'survive_crisis', 'carry_news', 'keep_order')
LEVELS = ('low', 'medium', 'high')
VOICES = ('baker', 'scribe', 'farmer', 'smith', 'shepherd', 'miller', 'resident')


def bridged(row, rng):
    """A plain account presented under a mind context, answered the plain way.

    The mind corpus only covers the kinds the simulation actually gossips
    about -- 21 of 139 -- so a character with a mind who meets any other kind
    has never been seen in training and collapses. The replay corpus carries
    44 kinds, and wrapping those rows in a mind block teaches the model that
    the context does not change what it knows how to say."""
    return {**row, 'voice': rng.choice(VOICES), 'control': 'say', 'act': 'bridge',
            'mind': {'goal': rng.choice(GOALS), 'stress': rng.choice(LEVELS),
                     'courage': rng.choice(LEVELS), 'memories': [], 'thoughts': []}}


WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def vocabulary(*groups):
    """Every word the corpora ever write, plus the words of the accounts they
    are written from. Anything outside it is the model inventing spelling."""
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


def quality(row, text, stopped, rules, known=None):
    """Exact match alone cannot tell a fair paraphrase from a collapse, so the
    failures the last run shipped get their own counters. Repetition is only
    one shape of collapse: short invented words ('It wellerve bea fim') carry
    no repeat and slipped through until the vocabulary check was added."""
    leaked = '# ' in text
    words = WORD.findall(text)
    # A reply may repeat anything its own prompt gave it -- place and actor
    # names differ in every row -- so the row's own words count as known.
    local = (vocabulary([row]) | known) if known else None
    invented = [w for w in words if w.lower() not in local] if local else []
    degenerate = bool(REPEAT.search(text)) or (len(text.split()) > 3 and
                  len(set(text.split())) < len(text.split()) / 2) or len(invented) >= 2
    copies = [span['text'] for span in row.get('copies', [])]
    return {'exact': stopped and text in approved_response(row, rules[row['rule']]),
            'stopped': stopped, 'leaked': leaked, 'degenerate': degenerate,
            'invented': invented[:4], 'empty': not text.strip(),
            'copied': all(span in text for span in copies) if copies else None}


def evaluate(model, tokenizer, rows, rules, label, known=None):
    """Runs on an already-CPU model. Moving the training model across devices
    mid-run strands the optimiser state on the accelerator and thrashes its
    allocator, which costs far more than the generation itself."""
    model.eval()
    scored = []
    for row in rows:
        result = generate(model, tokenizer, encoded(tokenizer, row))
        scored.append({'id': row['id'], 'act': row.get('act', 'account'),
                       'reference': row['output'], 'text': result['text'],
                       **quality(row, result['text'], result['stopped'], rules, known)})
    copied = [x['copied'] for x in scored if x['copied'] is not None]
    report = {'label': label, 'count': len(scored),
              'exact': sum(x['exact'] for x in scored),
              'stopped': sum(x['stopped'] for x in scored),
              'leaked': sum(x['leaked'] for x in scored),
              'degenerate': sum(x['degenerate'] for x in scored),
              'empty': sum(x['empty'] for x in scored),
              'copied': f'{sum(copied)}/{len(copied)}' if copied else 'n/a',
              'acts': {act: {'count': sum(x['act'] == act for x in scored),
                             'exact': sum(x['act'] == act and x['exact'] for x in scored),
                             'degenerate': sum(x['act'] == act and x['degenerate'] for x in scored)}
                       for act in sorted({x['act'] for x in scored})}}
    return report, scored


def loss_over(model, records, device, size=16):
    with torch.no_grad():
        values = [model.loss(batch(records[i:i + size], device)).item()
                  for i in range(0, len(records), size)]
    return sum(values) / len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Mind corpus directory')
    p.add_argument('--legacy', type=Path, required=True, help='Plain account corpus directory')
    p.add_argument('--chat', type=Path, help='Multi-turn conversation rows the guard also protects')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=Path, default=Path('models/crownless-core-v2/core.ccv2'))
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--replay-ratio', type=float, default=0.30)
    p.add_argument('--bridge-ratio', type=float, default=0.15,
                   help='Share of the batch that is a replayed account under a mind context')
    p.add_argument('--distill', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--guard-rows', type=int, default=48,
                   help='Held-out plain accounts the gate regenerates each check')
    p.add_argument('--guard-every', type=int, default=500)
    p.add_argument('--slack', type=int, default=2,
                   help='Exact plain-account answers allowed to be lost against the base')
    p.add_argument('--eval-rows', type=int, default=400)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--device', default='mps')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed)

    model, metadata = load_export(args.base, args.tokenizer, args.device)
    model.mode = 'conversation'
    assert sum(x.numel() for x in model.parameters()) <= 5000000
    teacher, _ = load_export(args.base, args.tokenizer, args.device)
    teacher.mode = 'conversation'; teacher.eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    if sha(args.data / 'rules.json') != metadata['rules_sha256']: p.error('Grammar differs')
    meanings = metadata['meaning_ids']

    bases = {s: read(args.data / f'{s}.jsonl') for s in ('train', 'validation', 'test')}
    for values in bases.values():
        for row in values: row['kind_id'] = meanings[row['rule']]
    mind = {s: build_rows(values, rules, f'{args.seed}:{s}', repeats=2 if s == 'train' else 1)
            for s, values in bases.items()}
    legacy = {s: legacy_rows(args.legacy / f'{s}.jsonl', meanings)
              for s in ('train', 'validation', 'test')}

    args.output.mkdir(parents=True)
    (args.output / 'tokenizer.json').write_bytes(args.tokenizer.read_bytes())
    manifest = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_hashes': {name: sha(Path(__file__).with_name(name)) for name in
                     ['crownless_v2.py', 'crownless_conversation.py', 'train_crownless_mind.py']},
                'base_sha256': sha(args.base), 'tokenizer_sha256': sha(args.tokenizer),
                'mind_manifest_sha256': sha(args.data / 'manifest.json'),
                'legacy_manifest_sha256': sha(args.legacy / 'manifest.json'),
                'rows': {'mind_train': len(mind['train']), 'legacy_train': len(legacy['train'])},
                'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
                'replay_ratio': args.replay_ratio, 'bridge_ratio': args.bridge_ratio,
                'distill': args.distill, 'lr': args.lr,
                'guard_rows': args.guard_rows, 'slack': args.slack,
                'scope': 'Mind context mixed with replayed plain accounts, anchored to the base '
                         'model by distillation and gated on plain-account validation.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    # Encoding all 50k rows up front costs several gigabytes of Python lists
    # and buys nothing: each step only looks at a batch. The validation sets
    # are small and fixed, so those stay pre-encoded.
    training, replay = mind['train'], legacy['train']
    mind_validation = [encoded(tokenizer, r) for r in mind['validation'][:244]]
    legacy_validation = [encoded(tokenizer, r) for r in legacy['validation'][:244]]

    # The gate compares against the model we started from, measured the same
    # way. Loss is only a proxy for the thing that broke last time, so the
    # guard reads the base model's actual answers on held-out plain accounts
    # and holds every checkpoint to them.
    legacy_base = loss_over(teacher, legacy_validation, args.device)
    mind_base = loss_over(teacher, mind_validation, args.device)
    # The guard has to cover every shape the runtime asks for. Plain accounts
    # alone missed that accumulating dialogue history is its own behaviour: a
    # model can answer one account perfectly and still fall apart by turn four.
    guard = legacy['validation'][:args.guard_rows]
    chat = [{**row, 'kind_id': meanings[row['rule']]}
            for row in read(args.chat)] if args.chat else []
    # A separate CPU copy reads the checkpoints, so neither the student nor the
    # teacher ever leaves the training device.
    known = vocabulary(*mind.values(), *legacy.values(), chat)
    scout, _ = load_export(args.base, args.tokenizer, 'cpu')
    scout.mode = 'conversation'
    base_report, _ = evaluate(scout, tokenizer, guard, rules, 'base', known)
    floor = base_report['exact'] - args.slack
    chat_report, _ = evaluate(scout, tokenizer, chat, rules, 'chat', known) if chat else (None, None)
    chat_floor = chat_report['exact'] - args.slack if chat_report else 0
    print(json.dumps({'base_legacy_validation': legacy_base, 'base_mind_validation': mind_base,
                      'base_guard': base_report, 'exact_floor': floor,
                      'base_chat': chat_report, 'chat_floor': chat_floor}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = random.Random(args.seed)
    replay_count = max(1, round(args.batch_size * args.replay_ratio))
    bridge_count = max(1, round(args.batch_size * args.bridge_ratio))
    best, best_step, history = float('inf'), 0, []
    candidate = float('inf')
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        mind_batch = [encoded(tokenizer, r) for r in
                      rng.choices(training, k=args.batch_size - replay_count - bridge_count)]
        mind_batch += [encoded(tokenizer, bridged(r, rng)) for r in rng.choices(replay, k=bridge_count)]
        replay_batch = [encoded(tokenizer, r) for r in rng.choices(replay, k=replay_count)]
        for group in optimizer.param_groups:
            group['lr'] = args.lr * min(step / 100, 1) * (.1 + .9 * (1 - step / args.steps))
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch(mind_batch + replay_batch, args.device))
        # Distil on the replayed rows only: those are the prompts the frozen
        # base actually answers well, so its logits are worth matching.
        anchor = batch(replay_batch, args.device)
        student_hidden, _ = model.hidden(anchor['tokens'], anchor['meta'])
        student_logits, _, _ = model.heads(student_hidden, anchor['candidates'], anchor['candidate_mask'])
        with torch.no_grad():
            teacher_hidden, _ = teacher.hidden(anchor['tokens'], anchor['meta'])
            teacher_logits, _, _ = teacher.heads(teacher_hidden, anchor['candidates'], anchor['candidate_mask'])
        # Only the answer positions carry the behaviour worth preserving;
        # averaging over the prompt as well dilutes the anchor with tokens the
        # task loss ignores.
        scored = (anchor['copy_labels'] != -100).flatten()
        distill = F.kl_div(F.log_softmax(student_logits.flatten(0, 1)[scored], -1),
                           F.softmax(teacher_logits.flatten(0, 1)[scored], -1),
                           reduction='batchmean')
        total = loss + args.distill * distill
        if not torch.isfinite(total): raise ValueError('Non-finite loss')
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            mind_val = loss_over(model, mind_validation, args.device)
            legacy_val = loss_over(model, legacy_validation, args.device)
            item = {'step': step, 'loss': loss.item(), 'distill': distill.item(),
                    'mind_validation': mind_val, 'legacy_validation': legacy_val,
                    'seconds': time.monotonic() - start}
            accepted = False
            if mind_val < best and step % args.guard_every == 0:
                scout.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
                report, _ = evaluate(scout, tokenizer, guard, rules, 'guard', known)
                accepted = report['exact'] >= floor and not report['degenerate'] and not report['leaked']
                item['guard'] = {k: report[k] for k in ('exact', 'degenerate', 'leaked', 'empty')}
                if chat:
                    # Dialogue cannot be held to the recorded wording: a reply
                    # can differ from the transcript and still be a good reply.
                    # What it may not do is stop early or invent words, which
                    # the vocabulary check now catches.
                    turns, _ = evaluate(scout, tokenizer, chat, rules, 'chat', known)
                    accepted = (accepted and not turns['degenerate'] and not turns['leaked']
                                and turns['stopped'] == turns['count'])
                    item['chat'] = {k: turns[k] for k in ('exact', 'stopped', 'degenerate', 'leaked')}
            item['accepted'] = accepted
            item['seconds'] = time.monotonic() - start
            if args.device == 'mps': torch.mps.empty_cache()
            history.append(item); print(json.dumps(item), flush=True)
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            # A run that never clears the gate costs the same hours as one that
            # does, and its weights are what the next decision reads. Keep the
            # lowest-validation checkpoint whichever way the gate went; best.pt
            # stays the gated one, so nothing downstream can confuse the two.
            if mind_val < candidate:
                candidate = mind_val
                torch.save({'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'step': step, 'validation': mind_val, 'accepted': accepted,
                            'guard': item.get('guard'), 'chat': item.get('chat')},
                           args.output / 'candidate.pt')
            if accepted:
                best, best_step = mind_val, step
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
            report = {'rejected': True, 'reason': 'No checkpoint held the plain account within tolerance',
                      'candidate_step': held['step'], 'validation': held['validation'],
                      'guard': held.get('guard'), 'chat': held.get('chat'),
                      'model_sha256': sha(args.output / 'rejected.ccv2')}
            (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report), flush=True)
        raise SystemExit('No checkpoint held the plain account within tolerance')

    saved = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    model.to('cpu').load_state_dict(saved['state'])
    export(model, args.tokenizer, args.output / 'core.ccv2',
           {k: metadata[k] for k in ['meaning_ids', 'kind_ids', 'rules_sha256']} |
           {'step': saved['step'], 'seed': args.seed})
    compact, _ = load_export(args.output / 'core.ccv2', args.tokenizer)

    reports = {}
    bridge_rng = random.Random(args.seed + 1)
    for label, rows in (('mind', mind['test'][:args.eval_rows]),
                        ('legacy', legacy['test'][:args.eval_rows]),
                        ('bridge', [bridged(r, bridge_rng) for r in legacy['test'][:args.eval_rows]]),
                        ('chat', chat)):
        if not rows: continue
        report, scored = evaluate(compact, tokenizer, rows, rules, label, known)
        (args.output / f'{label}-results.json').write_text(
            json.dumps({'report': report, 'rows': scored}, indent=2) + '\n')
        reports[label] = report
        print(json.dumps(report), flush=True)
    summary = {'best_step': saved['step'], 'base_legacy_validation': legacy_base,
               'model_sha256': sha(args.output / 'core.ccv2'), 'reports': reports}
    (args.output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'model_sha256': summary['model_sha256'], 'best_step': saved['step']}), flush=True)


if __name__ == '__main__': main()
