"""Run the four registered pilot arms, then score their selected checkpoints."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from canada_narrative import prepare, read_json, write_json, digest, EXPERIMENT


def run(delivery, data, output):
    import numpy as np
    import torch
    from run_canada_narrative import train, setup, evaluate, verify_code
    from review_canada_narrative import validate_pair
    verify_code()
    prepare(delivery, data)
    manifest_sha = digest(data / 'manifest.json')
    if manifest_sha != read_json(EXPERIMENT / 'preparation.json')['manifest_sha256']:
        raise ValueError('Rebuilt training streams differ from the accepted preparation')
    for comparison, arm in [('AB', 'A'), ('AB', 'B'), ('BC', 'B'), ('BC', 'C')]:
        train(SimpleNamespace(data=data, output=output / f'{comparison}-{arm}',
                              comparison=comparison, arm=arm, seed=7, device='cuda', pilot_decision=None))
    validation = np.memmap(data / 'validation.bin', dtype='<u2', mode='r')
    starts = read_json(data / 'evaluation.json')['outcome']
    lengths = np.array(read_json(data / 'token_bytes.json'))
    pairs = {}
    for comparison, arms in [('AB', ['A', 'B']), ('BC', ['B', 'C'])]:
        records = [read_json(output / f'{comparison}-{arm}' / 'result.json') for arm in arms]
        validate_pair(*records, comparison, manifest_sha)
        scores = []
        for arm, record in zip(arms, records):
            checkpoint = output / f'{comparison}-{arm}' / 'best.pt'
            if digest(checkpoint) != record['best_sha256']:
                raise ValueError('Selected checkpoint changed')
            model, optimizer = setup(7, 'cuda'); del optimizer
            state = torch.load(checkpoint, map_location='cuda', weights_only=True)
            if state['step'] != record['selected_step']:
                raise ValueError('Checkpoint selection differs')
            model.load_state_dict(state['model'])
            scores.append(evaluate(model, validation, starts, lengths, 'cuda'))
            del model, state
        pairs[comparison] = {'runs': records, 'outcome_scores': scores,
            'relative_loss_improvement': 1 - scores[1]['bits_per_byte'] / scores[0]['bits_per_byte']}
    write_json(output / 'pilot.json', {'status': 'passed', 'meaning': 'Training and paired scoring completed',
        'pairs': pairs, 'human_review': 'pending', 'generation_and_repetition_review': 'pending',
        'manifest_sha256': manifest_sha})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.delivery, args.data, args.output)
