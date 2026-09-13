"""Run the registered pilot, or time synthetic updates on the selected host."""
import argparse
import contextlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from canada_narrative import EXPERIMENT, ROOT, digest, read_json, write_json, verify_prepared
from subword_model import CONFIGS, create


def verify_code():
    from importlib.metadata import version
    lock = read_json(EXPERIMENT / 'implementation.lock.json')
    for name, expected in lock['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Implementation identity mismatch: {name}')
    for name, expected in lock['packages'].items():
        if version(name).split('+')[0] != expected:
            raise ValueError(f'Package version mismatch: {name}')


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()


def setup(seed, device):
    if device == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ValueError('Use a CUDA device with bfloat16 support')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = create(CONFIGS['5m-1024'], seed).to(device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if p.ndim == 2], 'weight_decay': .01},
        {'params': [p for p in model.parameters() if p.ndim == 1], 'weight_decay': 0.}],
        lr=.0003, betas=(.9, .999), eps=1e-8)
    return model, optimizer


def state_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def windows(data, offset, count, context):
    """Each scored target is the next cyclic stream position, including tails."""
    for begin in range(0, count, context):
        valid = min(context, count - begin)
        indices = (np.arange(context) + offset + begin) % len(data)
        x = np.asarray(data[(indices - 1) % len(data)], dtype=np.int64).copy()
        y = np.asarray(data[indices], dtype=np.int64).copy()
        y[valid:] = -100
        yield x, y, valid


def update(model, optimizer, data, offset, count, lr, device):
    model.train(); optimizer.zero_grad(set_to_none=True); total = 0.
    for x, y, valid in windows(data, offset, count, model.context):
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        y = torch.from_numpy(y).to(device)
        with amp(device):
            loss = F.cross_entropy(model(x, dropout=.1)[0].float(), y, reduction='sum') / count
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward(); total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    return total


@torch.no_grad()
def evaluate(model, data, starts, lengths, device):
    model.eval(); total = 0.; byte_count = 0
    for start in starts:
        target = np.asarray(data[start:start + 256], dtype=np.int64)
        window = np.asarray(data[start - 769:start + 255], dtype=np.int64)
        if len(target) != 256 or len(window) != 1024:
            raise ValueError('Evaluation window outside stream')
        with amp(device):
            logits = model(torch.tensor(window, device=device).unsqueeze(0))[0, -256:].float()
            loss = F.cross_entropy(logits, torch.tensor(target, device=device), reduction='sum')
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite evaluation loss')
        total += loss.item(); byte_count += int(lengths[target].sum())
    return {'bits_per_byte': total / math.log(2) / byte_count,
            'tokens': len(starts) * 256, 'bytes': byte_count}


def learning_rate(step, steps):
    warmup = min(2000, max(1, steps // 50))
    value = .0003 * min(1., step / warmup)
    if step > warmup:
        value *= .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    return value


def train(args):
    manifest = verify_prepared(args.data)
    receipt = read_json(EXPERIMENT / 'preparation.json')
    if digest(args.data / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Prepared streams differ from the registered preparation')
    contract = read_json(EXPERIMENT / 'contract.json')
    comparison = contract['comparisons'][args.comparison]
    if args.arm not in comparison['arms']:
        raise ValueError('Arm belongs to a different comparison')
    budget = comparison['target_tokens_per_arm']; block = contract['training']['tokens_per_update']
    args.output.mkdir(parents=True, exist_ok=False)
    data = np.memmap(args.data / f'{args.arm}.bin', dtype='<u2', mode='r')
    validation = np.memmap(args.data / 'validation.bin', dtype='<u2', mode='r')
    evaluation = read_json(args.data / 'evaluation.json')
    lengths = np.array(read_json(args.data / 'token_bytes.json'))
    seed = contract['pilot_seed']; model, optimizer = setup(seed, args.device)
    initial = state_digest(model); best = math.inf; history = []; offset = 0
    steps = math.ceil(budget / block); started = time.monotonic()
    identity = {'comparison': args.comparison, 'arm': args.arm, 'seed': seed,
                'device': args.device, 'platform': platform.platform(),
                'config': CONFIGS['5m-1024'], 'parameters': sum(p.numel() for p in model.parameters()),
                'initial_weights_sha256': initial,
                'contract_sha256': digest(EXPERIMENT / 'contract.json'),
                'implementation_sha256': digest(EXPERIMENT / 'implementation.lock.json'),
                'manifest_sha256': digest(args.data / 'manifest.json')}
    write_json(args.output / 'identity.json', identity)
    for step in range(1, steps + 1):
        count = min(block, budget - offset)
        loss = update(model, optimizer, data, offset, count, learning_rate(step, steps), args.device)
        offset += count
        if step % 200 == 0 or step == steps:
            score = evaluate(model, validation, evaluation['selection'], lengths, args.device)
            row = {'step': step, 'target_presentations': offset, 'last_update_loss': loss,
                   'selection': score, 'elapsed_seconds': time.monotonic() - started}
            history.append(row); write_json(args.output / 'history.json', history)
            state = {'config': CONFIGS['5m-1024'], 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'step': step, 'target_presentations': offset,
                     'cpu_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}
            torch.save(state, args.output / 'last.pt')
            if score['bits_per_byte'] < best:
                best = score['bits_per_byte']; torch.save(state, args.output / 'best.pt')
                selected = step
            print(json.dumps(row), flush=True)
    write_json(args.output / 'result.json', {**identity, 'status': 'trained', 'selected_step': selected,
        'target_presentations': offset, 'stream_tokens': len(data),
        'full_target_passes': offset // len(data), 'tail_targets': offset % len(data),
        'best_sha256': digest(args.output / 'best.pt'), 'elapsed_seconds': time.monotonic() - started})


def benchmark(args):
    model, optimizer = setup(7, args.device)
    synthetic = np.random.default_rng(107).integers(0, 2048, 32768, dtype=np.uint16)
    update(model, optimizer, synthetic, 0, 8192, .0003, args.device)
    times = []
    for step in range(3):
        start = time.monotonic()
        update(model, optimizer, synthetic, step * 8192, 8192, .0003, args.device)
        times.append(time.monotonic() - start)
    validation_start = time.monotonic()
    evaluate(model, synthetic, [1024, 2048, 3072, 4096], np.ones(2048), args.device)
    window_seconds = (time.monotonic() - validation_start) / 4
    per_update = float(np.median(times)); projections = {}
    for name, comparison in read_json(EXPERIMENT / 'contract.json')['comparisons'].items():
        steps = math.ceil(comparison['target_tokens_per_arm'] / 8192)
        seconds = 2 * (steps * per_update + math.ceil(steps / 200) * 64 * window_seconds)
        projections[name] = {'paired_training_and_selection_hours': seconds / 3600,
                             'planning_hours_with_30_percent_margin': seconds * 1.3 / 3600}
    result = {'device': args.device, 'platform': platform.platform(),
              'parameters': sum(p.numel() for p in model.parameters()),
              'synthetic_updates': 4, 'corpus_training_presentations': 0,
              'measured_update_seconds': times, 'median_update_seconds': per_update,
              'evaluation_window_seconds': window_seconds, 'projections': projections,
              'scope': 'Local synthetic timing; checkpoint I/O, final scoring and generation are additional',
              'cloud_cost': 'Pending a target host and measured timing on that host'}
    write_json(args.output, result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    timing = sub.add_parser('benchmark'); timing.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('train')
    run.add_argument('--data', type=Path, required=True); run.add_argument('--output', type=Path, required=True)
    run.add_argument('--comparison', choices=['AB', 'BC'], required=True)
    run.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    for child in [timing, run]:
        child.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(); verify_code()
    (benchmark if args.command == 'benchmark' else train)(args)
