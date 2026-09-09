"""Tune ZERO on Crownless v2 event lists, with loss on the spoken line only."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from zero_torch import load, write, update


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(root, split, context):
    path = root / (split + '.txt')
    data = path.read_bytes()
    rows = [json.loads(line) for line in (root / (split + '.audit.jsonl')).read_text().splitlines()]
    examples = []
    cursor = 0
    for row in rows:
        start, size, output = row['text_start'], row['text_bytes'], row['output_start']
        if start != cursor or not start < output < start + size:
            raise ValueError('Invalid audit offsets')
        sample = data[start:start + size]
        target = (row['output'] + '\n').encode('ascii')
        prefix = data[start:output]
        if sample != prefix + target + b'\n' or not prefix.endswith(b'\n'):
            raise ValueError('Audit text mismatch')
        if any(c >= 128 or c == 0 for c in sample):
            raise ValueError('Expected ASCII text without padding bytes')
        dropped = 0
        while len(prefix + target) > context + 1 and prefix.count(b'\n') > 1:
            prefix = prefix.split(b'\n', 1)[1]
            dropped += 1
        if len(prefix + target) > context + 1:
            raise ValueError('Final event and speech exceed model context')
        if not all(line.startswith(b'- ') for line in prefix.splitlines()):
            raise ValueError('Expected plain event lines')
        examples.append(dict(id=row['id'], prefix=prefix, target=target,
                             dropped=dropped, kind=row['input']['kind'],
                             world=row.get('provenance', {}).get('world_seed')))
        cursor += size
    if cursor != len(data) or not examples:
        raise ValueError('Incomplete or empty split')
    return examples


def batch(examples, device):
    length = max(len(e['prefix']) + len(e['target']) - 1 for e in examples)
    x = torch.zeros((len(examples), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, e in enumerate(examples):
        seq = torch.tensor(list(e['prefix'] + e['target']), dtype=torch.long)
        n, first = len(seq) - 1, len(e['prefix']) - 1
        x[i, :n] = seq[:-1]
        y[i, first:n] = seq[first + 1:]
    return x.to(device), y.to(device)


@torch.no_grad()
def score(model, examples, device, batch_size):
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = batch(examples[i:i + batch_size], device)
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction='sum').item()
        count += (y != -100).sum().item()
    return total / count


@torch.no_grad()
def generate(model, examples, device, max_chars=160):
    model.eval()
    results = []
    for e in examples:
        tokens = list(e['prefix'])
        output = []
        stopped = False
        for _ in range(min(max_chars, model.context + 1 - len(tokens))):
            logits = model(torch.tensor([tokens], device=device))[0, -1]
            # ASCII printable characters and the speech-ending newline.
            logits[:10] = -torch.inf
            logits[11:32] = -torch.inf
            logits[127:] = -torch.inf
            token = logits.argmax().item()
            if token == 10:
                stopped = True
                break
            output.append(token)
            tokens.append(token)
        results.append(dict(id=e['id'], kind=e['kind'], events=e['prefix'].decode(),
                            expected=e['target'].decode().rstrip('\n'),
                            generated=bytes(output).decode(), stopped=stopped))
    return results


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-seconds', type=int, default=1200)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.lr, args.max_seconds) <= 0:
        p.error('Training bounds must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, _ = load(args.base)
    splits = {s: read_split(args.data, s, model.context)
              for s in ['train', 'validation', 'test', 'editorial_test']}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        for key in ['target', 'id', 'world']:
            if ({e[key] for e in splits[a] if e[key] is not None} &
                    {e[key] for e in splits[b] if e[key] is not None}):
                raise ValueError('Overlapping splits: ' + key)
    metadata = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    base_sha256=digest(args.base), base_step=model.header[10],
                    parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
                    source_sha256={name: digest(Path(__file__).parent / name)
                                   for name in ['tune_crownless.py', 'zero_torch.py']},
                    data_sha256={p.name: digest(p) for p in args.data.iterdir()
                                 if p.suffix in ['.txt', '.jsonl']},
                    splits={s: dict(rows=len(es), dropped_events=sum(e['dropped'] for e in es))
                            for s, es in splits.items()}, optimizer='fresh Adam moments')
    save_json(args.output / 'metadata.json', metadata)
    model.to(args.device)
    moments = [(torch.zeros_like(w), torch.zeros_like(w)) for w in model.weights]
    # Fixed validation examples cover each kind. Test sets are opened for scoring only after selection.
    selected = {}
    for e in splits['validation']:
        selected.setdefault(e['kind'], e)
    start = time.monotonic()
    base_loss = score(model, splits['validation'], args.device, args.batch_size)
    before = generate(model, list(selected.values()), args.device)
    save_json(args.output / 'before.json', before)
    best, best_step = base_loss, 0
    write(args.output / 'best.ckpt', model, moments, 0)
    history = [dict(step=0, validation_loss=base_loss, elapsed=time.monotonic() - start)]
    print(json.dumps(history[-1]), flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(rng.choices(splits['train'], k=args.batch_size), args.device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, dropout=0.05).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        update(model, moments, step, args.lr)
        deadline = time.monotonic() - start >= args.max_seconds
        if step % args.eval_every == 0 or step == args.steps or deadline:
            val = score(model, splits['validation'], args.device, args.batch_size)
            history.append(dict(step=step, validation_loss=val, train_batch_loss=loss.item(),
                                elapsed=time.monotonic() - start))
            save_json(args.output / 'history.json', history)
            print(json.dumps(history[-1]), flush=True)
            if val < best:
                best, best_step = val, step
                write(args.output / 'best.ckpt', model, moments, step)
            if deadline:
                break
    write(args.output / 'last.ckpt', model, moments, step)
    del model, moments
    model, _ = load(args.output / 'best.ckpt')
    model.to(args.device)
    after = generate(model, list(selected.values()), args.device)
    editorial = generate(model, splits['editorial_test'], args.device)
    save_json(args.output / 'after.json', after)
    save_json(args.output / 'editorial.json', editorial)
    result = dict(parameters=metadata['parameters'], updates=step, selected_update=best_step,
                  base_validation_loss=base_loss, validation_loss=best,
                  test_loss=score(model, splits['test'], args.device, args.batch_size),
                  editorial_loss=score(model, splits['editorial_test'], args.device, args.batch_size),
                  elapsed_seconds=time.monotonic() - start,
                  validation_sample_stops=sum(e['stopped'] for e in after),
                  validation_sample_count=len(after),
                  editorial_stops=sum(e['stopped'] for e in editorial),
                  editorial_count=len(editorial),
                  evaluation='Greedy ASCII decoding, 160 character cap; loss is per target character')
    save_json(args.output / 'result.json', result)
    print(json.dumps(result), flush=True)
    (args.output / 'SHA256SUMS').write_text(''.join(
        digest(p) + '  ' + p.name + '\n' for p in sorted(args.output.iterdir()) if p.is_file()))


if __name__ == '__main__':
    main()
