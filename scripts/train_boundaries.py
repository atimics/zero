"""Train the four 5M record-boundary arms with identical target rosters."""
import argparse
import itertools
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from canada_narrative import digest, read_json, verify_prepared
from boundary_common import contract, write, EXPERIMENT
from boundary_data import roster, roster_digest, load_records, packs, forward
from run_canada_narrative import setup, state_digest, evaluate, amp, learning_rate


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary); temporary.replace(path)


def train_core(model, optimizer, data, records, visits, output, config, run_identity, score, device):
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'identity.json'
    if identity_path.exists():
        if read_json(identity_path) != run_identity:
            raise ValueError('Resume identity differs')
    elif any(output.iterdir()):
        raise ValueError('Output has files without an identity')
    else:
        write(identity_path, run_identity)
    if (output / 'result.json').exists():
        result = read_json(output / 'result.json')
        if result['identity'] != run_identity or digest(output / 'best.pt') != result['best_sha256']:
            raise ValueError('Completed result identity differs')
        return result
    windows_per_update = config['windows_per_update']; microbatch = config['microbatch_windows']
    expected = sum(n for _, n in visits)
    total_windows = math.ceil((expected + len(visits)) / model.context)
    steps = math.ceil(total_windows / windows_per_update)
    step = 0; consumed = 0; best = math.inf; selected = None; history = []; elapsed = 0.
    accounted = {'forward_tokens': 0, 'masked_tokens': 0, 'padding_tokens': 0}
    state_path = output / 'last.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state['identity'] != run_identity:
            raise ValueError('Checkpoint identity differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        step = state['step']; consumed = state['consumed']; best = state['best']
        selected = state['selected']; history = state['history']; elapsed = state['elapsed_seconds']
        accounted = state['accounted']
        # best weights travel inside the atomic resume checkpoint, including after crashes.
        save_state(output / 'best.pt', state['best_checkpoint'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
    rows = iter(packs(data, records, visits, model.context))
    for _ in range(min(step * windows_per_update, total_windows)):
        next(rows)
    started = time.monotonic()
    while step < steps:
        group = list(itertools.islice(rows, windows_per_update))
        count = sum(int((row[1] != -100).sum()) for row in group)
        model.train(); optimizer.zero_grad(set_to_none=True); loss_value = 0.
        for begin in range(0, len(group), microbatch):
            part = group[begin:begin+microbatch]
            x, y, segments = [torch.from_numpy(np.stack([r[i] for r in part])).to(device) for i in range(3)]
            if count:
                with amp(device):
                    logits = forward(model, x, segments, run_identity['isolate'], config['dropout'])
                    loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction='sum') / count
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); loss_value += loss.item()
        lr = learning_rate(step + 1, steps)
        grad = 0.
        if count:
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] = lr
            optimizer.step()
        step += 1; consumed += count
        accounted['forward_tokens'] += len(group) * model.context
        accounted['masked_tokens'] += sum(int((r[1] == -100).sum()) for r in group)
        accounted['padding_tokens'] += sum(int((r[2] == -1).sum()) for r in group)
        if step % config['selection_every'] == 0 or step == steps:
            selection = score(model)
            if not math.isfinite(selection['bits_per_byte']):
                raise ValueError('Nonfinite selection loss')
            row = {'step': step, 'target_presentations': consumed, 'loss': loss_value,
                   'gradient_norm_before_clip': grad, 'lr': lr, 'selection': selection,
                   'elapsed_seconds': elapsed + time.monotonic() - started, **accounted}
            history.append(row)
            if selection['bits_per_byte'] < best:
                best = selection['bits_per_byte']; selected = step
                best_checkpoint = {'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                   'step': step, 'identity': run_identity}
            else:
                best_checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            state = {'identity': run_identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'step': step, 'consumed': consumed, 'best': best, 'selected': selected,
                     'history': history, 'elapsed_seconds': row['elapsed_seconds'], 'accounted': accounted,
                     'cpu_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                     'best_checkpoint': best_checkpoint}
            save_state(state_path, state)
            save_state(output / 'best.pt', best_checkpoint)
            write(output / 'history.json', history)
            print(row, flush=True)
    if consumed != expected:
        raise ValueError('Scored target budget mismatch')
    result = {'status': 'trained', 'identity': run_identity, 'target_presentations': consumed,
              'selected_step': selected, 'best_selection': best, 'best_sha256': digest(output / 'best.pt'),
              'elapsed_seconds': elapsed + time.monotonic() - started, **accounted}
    write(output / 'result.json', result)
    return result


def train(data_dir, output, arm, device='cuda'):
    config = contract(); verify_prepared(data_dir)
    from canada_narrative import EXPERIMENT as CANADA
    if digest(data_dir / 'manifest.json') != read_json(CANADA / 'preparation.json')['manifest_sha256']:
        raise ValueError('Prepared data differs from the accepted receipt')
    if device not in ['cpu', 'cuda']:
        raise ValueError('This experiment supports CPU diagnostics and CUDA training')
    shuffled, isolated = config['arms'][arm]
    data = np.memmap(data_dir / 'B.bin', dtype='<u2', mode='r')
    records = load_records(data_dir / 'B.records.jsonl', len(data))
    visits = roster(records, config['target_presentations'], shuffled, config['seed'])
    model, optimizer = setup(config['seed'], device)
    if sum(p.numel() for p in model.parameters()) != config['parameters']:
        raise ValueError('Model size differs')
    run_identity = {'arm': arm, 'shuffle': shuffled, 'isolate': isolated, 'seed': config['seed'],
                    'device': device, 'platform': platform.platform(), 'torch_version': str(torch.__version__),
                    'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'cpu',
                    'initial_weights_sha256': state_digest(model),
                    'contract_sha256': digest(EXPERIMENT / 'contract.json'),
                    'source_lock_sha256': digest(EXPERIMENT / 'source.lock.json'),
                    'data_manifest_sha256': digest(data_dir / 'manifest.json'),
                    'ordered_roster_sha256': roster_digest(visits),
                    'target_multiset_sha256': roster_digest(visits, canonical=True)}
    validation = np.memmap(data_dir / 'validation.bin', dtype='<u2', mode='r')
    lengths = np.array(read_json(data_dir / 'token_bytes.json'))
    starts = read_json(data_dir / 'evaluation.json')['selection']
    return train_core(model, optimizer, data, records, visits, output, config, run_identity,
                      lambda m: evaluate(m, validation, starts, lengths, device), device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['P0','P1','P2','P3'], required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args(); train(args.data, args.output, args.arm, args.device)
