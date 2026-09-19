"""Generate the registered writing comparisons from the completed pilot."""
import argparse
from pathlib import Path
from types import SimpleNamespace
from canada_narrative import prepare, digest, read_json, write_json, EXPERIMENT


def run(delivery, data, checkpoints, output):
    from review_canada_narrative import build
    from run_canada_narrative import verify_code
    verify_code(); prepare(delivery, data)
    if digest(data / 'manifest.json') != read_json(EXPERIMENT / 'preparation.json')['manifest_sha256']:
        raise ValueError('Prepared streams differ from the completed pilot')
    metrics = {}
    for comparison, control, candidate in [('AB', 'A', 'B'), ('BC', 'B', 'C')]:
        print('Generating ' + comparison + ': 200 matched pairs', flush=True)
        directory = output / comparison
        build(SimpleNamespace(data=data, comparison=comparison,
            control=checkpoints / f'{comparison}-{control}',
            candidate=checkpoints / f'{comparison}-{candidate}', output=directory))
        metrics[comparison] = read_json(directory / 'metrics.json')
        print('Completed ' + comparison, flush=True)
    write_json(output / 'samples.json', {'status': 'passed', 'pairs_per_comparison': 200,
        'total_continuations': 800, 'additional_training_presentations': 0,
        'metrics': metrics, 'human_review': 'pending'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['delivery', 'data', 'checkpoints', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); run(a.delivery, a.data, a.checkpoints, a.output)
