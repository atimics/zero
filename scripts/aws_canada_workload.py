"""Measure the frozen CUDA training update using synthetic token IDs."""
import argparse
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from canada_narrative import EXPERIMENT, digest, read_json, write_json
from run_canada_narrative import setup, update, evaluate, synchronize, verify_code, amp


def run(output):
    verify_code()
    config = read_json(EXPERIMENT / 'contract.json')
    data = np.random.default_rng(107).integers(0, 2048, 32768, dtype=np.uint16)
    cpu, cpu_optimizer = setup(7, 'cpu')
    gpu, optimizer = setup(7, 'cuda')
    cpu.eval(); gpu.eval()
    with torch.no_grad():
        x = torch.tensor(data[:1024].astype('int64'))[None]
        reference = cpu(x)
        with amp('cuda'):
            observed = gpu(x.cuda()).float().cpu()
        difference = (reference - observed).abs().max().item()
        if not math.isfinite(difference) or difference > .05:
            raise ValueError('CUDA bfloat16 logit check exceeded 0.05')
    del cpu, cpu_optimizer, reference, observed
    micro = config['training']['microbatch_sequences']
    for step in range(5):
        update(gpu, optimizer, data, step * 8192, 8192, .0003, 'cuda', micro)
    synchronize('cuda'); times = []; losses = []
    for step in range(100):
        started = time.monotonic()
        losses.append(update(gpu, optimizer, data, step * 8192, 8192, .0003, 'cuda', micro))
        synchronize('cuda'); times.append(time.monotonic() - started)
    started = time.monotonic()
    evaluate(gpu, data, list(range(1024, 17408, 256)), np.ones(2048), 'cuda')
    window_seconds = (time.monotonic() - started) / 64
    median = statistics.median(times)
    projections = {}
    for name, spec in config['comparisons'].items():
        steps = math.ceil(spec['target_tokens_per_arm'] / 8192)
        hours = 2 * (steps * median + math.ceil(steps / 200) * 64 * window_seconds) / 3600
        projections[name] = {'paired_training_and_selection_hours': hours,
                             'hours_with_30_percent_margin': 1.3 * hours,
                             'instance_usd_with_margin': 1.3 * hours * .8936}
    result = {'status': 'passed', 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'platform': platform.platform(), 'contract_sha256': digest(EXPERIMENT / 'contract.json'),
              'implementation_sha256': digest(EXPERIMENT / 'implementation.lock.json'),
              'corpus_training_presentations': 0, 'synthetic_updates': 105,
              'logit_max_abs_difference_from_cpu': difference, 'logit_tolerance': .05,
              'update_seconds': times, 'median_update_seconds': median,
              'p95_update_seconds': float(np.percentile(times, 95)),
              'first_measured_loss': losses[0], 'last_measured_loss': losses[-1],
              'evaluation_window_seconds': window_seconds, 'projections': projections,
              'scope': 'Synthetic timing; checkpoint I/O, final scoring and generation add time'}
    write_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
