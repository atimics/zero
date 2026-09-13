"""Compare CPU/Apple GPU throughput and record full-model numerical checks."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from canada_narrative import write_json
from run_canada_narrative import setup, update, evaluate, synchronize, verify_code


def run(output):
    data = np.random.default_rng(107).integers(0, 2048, 32768, dtype=np.uint16)
    rows = []
    for device, threads, micro in [('cpu', 2, 1), ('cpu', 4, 1), ('cpu', 8, 1),
                                   ('cpu', 4, 4), ('cpu', 8, 4),
                                   ('mps', 2, 1), ('mps', 2, 4), ('mps', 2, 8)]:
        model, optimizer = setup(7, device, threads)
        update(model, optimizer, data, 0, 8192, .0003, device, micro)
        synchronize(device); times = []
        for step in range(3):
            start = time.monotonic()
            loss = update(model, optimizer, data, step * 8192, 8192, .0003, device, micro)
            synchronize(device); times.append(time.monotonic() - start)
        row = {'device': device, 'threads': threads, 'microbatch_sequences': micro,
               'seconds': times, 'median': float(np.median(times)), 'last_loss': loss}
        rows.append(row); print(json.dumps(row), flush=True)
        del model, optimizer; gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()

    def numerical(device, dropout):
        model, optimizer = setup(7, device)
        model.eval()
        with torch.no_grad():
            before = model(torch.tensor(data[:1024].astype('int64'), device=device)[None]).cpu().numpy()
        losses = [update(model, optimizer, data, step * 8192, 8192, .0003, device, 4, dropout)
                  for step in range(2)]
        weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
        score = evaluate(model, data, [2048, 4096, 8192, 12288], np.ones(2048), device)
        del model, optimizer; gc.collect()
        return before, weights, losses, score

    cpu = numerical('cpu', 0); gpu = numerical('mps', 0)
    first = numerical('mps', .1); second = numerical('mps', .1)
    checks = {
        'initial_logits_max_abs': float(np.max(np.abs(cpu[0] - gpu[0]))),
        'updated_weights_max_abs': float(np.max(np.abs(cpu[1] - gpu[1]))),
        'updated_weights_relative_l2': float(np.linalg.norm(cpu[1] - gpu[1]) / np.linalg.norm(cpu[1])),
        'loss_max_abs': float(np.max(np.abs(np.array(cpu[2]) - gpu[2]))),
        'evaluation_bpb_abs': abs(cpu[3]['bits_per_byte'] - gpu[3]['bits_per_byte']),
        'repeat_weights_max_abs': float(np.max(np.abs(first[1] - second[1]))),
        'repeat_loss_max_abs': float(np.max(np.abs(np.array(first[2]) - second[2])))
    }
    tolerances = {'initial_logits_max_abs': 1e-4, 'updated_weights_max_abs': 1e-4,
                  'updated_weights_relative_l2': 5e-6, 'loss_max_abs': 1e-5,
                  'evaluation_bpb_abs': 1e-5, 'repeat_weights_max_abs': 2e-6,
                  'repeat_loss_max_abs': 1e-6}
    result = {'schema': 1, 'parameters': 5049600, 'corpus_training_presentations': 0,
              'sweep': rows, 'numerical_checks': checks, 'engineering_tolerances': tolerances,
              'all_numerical_checks_pass': all(checks[k] <= v for k, v in tolerances.items()),
              'scope': 'Two synthetic updates check numerical agreement and short-run repeatability. Full training remains an experiment.'}
    write_json(output, result); print(json.dumps(result, indent=2))
    if not result['all_numerical_checks_pass']:
        raise ValueError('GPU numerical checks exceed the engineering tolerances')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify_code(); run(args.output)
