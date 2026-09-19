"""Freeze a bounded Oregon generation job using verified pilot checkpoints."""
import argparse
import tarfile
from pathlib import Path
from canada_narrative import ROOT, digest, read_json, write_json
from prepare_canada_pilot import prepare_pilot
from review_canada_narrative import validate_pair


def checkpoint_files(checkpoints):
    files = {}
    for comparison, arms in [('AB', ['A', 'B']), ('BC', ['B', 'C'])]:
        records = [read_json(checkpoints / f'{comparison}-{arm}/result.json') for arm in arms]
        validate_pair(*records, comparison, records[0]['manifest_sha256'])
        for arm, record in zip(arms, records):
            directory = f'{comparison}-{arm}'
            path = checkpoints / directory / 'best.pt'
            if digest(path) != record['best_sha256']:
                raise ValueError('Selected checkpoint identity mismatch: ' + directory)
            files[directory + '/best.pt'] = path
            files[directory + '/result.json'] = checkpoints / directory / 'result.json'
    return files


def prepare_samples(delivery, checkpoints, output):
    files = checkpoint_files(checkpoints)
    manifest = prepare_pilot(delivery, output, region='us-west-2')
    source = output / 'source.tar.gz'; previous = digest(source)
    old = output / 'pilot.tar.gz'; source.rename(old)
    with tarfile.open(source, 'w:gz') as dst, tarfile.open(old) as src:
        for entry in src.getmembers():
            dst.addfile(entry, src.extractfile(entry) if entry.isfile() else None)
        dst.add(ROOT / 'scripts/aws_canada_samples.py', arcname='scripts/aws_canada_samples.py')
        for name, path in files.items():
            dst.add(path, arcname='checkpoints/' + name)
    old.unlink()
    script = (output / 'user-data.template.sh').read_text().replace(previous, digest(source))
    script = script.replace('scripts/aws_canada_pilot.py --delivery delivery --data prepared --output output',
        'scripts/aws_canada_samples.py --delivery delivery --data prepared --checkpoints checkpoints --output output')
    (output / 'user-data.template.sh').write_text(script)
    manifest.update(scope='800 continuations from four selected checkpoints; registered 200-pair AB and BC packets',
        result_file='samples.json', additional_training_presentations=0,
        checkpoint_files={name: digest(path) for name, path in files.items()})
    manifest['files'] = {name: digest(output / name) for name in manifest['files']}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['delivery', 'checkpoints', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); prepare_samples(a.delivery, a.checkpoints, a.output)
