"""Package the Canada-only training pilot within a US$2 budget."""
import argparse
import tarfile
from pathlib import Path
from canada_narrative import ROOT, EXPERIMENT, checked, digest, read_json, write_json
from prepare_canada_aws import prepare


def delivery_files(delivery):
    lock = read_json(EXPERIMENT / 'inputs.lock.json')
    files = dict(lock['delivery'])
    for item in [*lock['baseline'].values(), lock['baseline_manifest']]:
        files[item['path']] = item['sha256']
    for release in lock['releases'].values():
        files.update({release['directory'] + '/' + name: sha for name, sha in release['files'].items()})
    return {name: checked(delivery, name, sha) for name, sha in files.items()}


def prepare_pilot(delivery, output):
    inputs = delivery_files(delivery)
    manifest = prepare(output, subnet=None, instance_type='g5.xlarge', region='ca-central-1')
    original_sha = digest(output / 'source.tar.gz')
    source = output / 'source.tar.gz'; old = output / 'code.tar.gz'; source.rename(old)
    with tarfile.open(source, 'w:gz') as dst, tarfile.open(old) as src:
        for entry in src.getmembers():
            dst.addfile(entry, src.extractfile(entry) if entry.isfile() else None)
        dst.add(ROOT / 'scripts/aws_canada_pilot.py', arcname='scripts/aws_canada_pilot.py')
        for name, path in sorted(inputs.items()):
            dst.add(path, arcname='delivery/' + name)
    old.unlink()
    script = (output / 'user-data.template.sh').read_text()
    script = script.replace(original_sha, digest(source)).replace('shutdown -h +30', 'shutdown -h +85')
    script = script.replace('timeout 900 "$PYTHON" scripts/aws_canada_workload.py --output output/timing.json',
        'timeout 4500 "$PYTHON" scripts/aws_canada_pilot.py --delivery delivery --data prepared --output output')
    (output / 'user-data.template.sh').write_text(script)
    stack = read_json(output / 'stack.json')
    code = stack['Resources']['Watchdog']['Properties']['Code']['ZipFile']
    stack['Resources']['Watchdog']['Properties']['Code']['ZipFile'] = code.replace('>= 1800', '>= 5100')
    write_json(output / 'stack.json', stack)
    manifest.update(scope='Four Canada corpus pilot arms, seed 7, then paired held-out loss scoring',
        workload_timeout_seconds=4500, watchdog_maximum_age_seconds=5100,
        result_file='pilot.json', controller_wait_seconds=5400,
        planning_instance_usd_at_86_minutes=1.117 * 86 / 60,
        shutdown='Guest shutdown at success/failure and +85 minutes; AWS watchdog at 85 minutes',
        corpus_location='Canada only', delivery_files={name: digest(path) for name, path in inputs.items()})
    manifest.pop('planning_instance_usd_at_31_minutes')
    manifest['files'] = {name: digest(output / name) for name in manifest['files']}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare_pilot(a.delivery, a.output)
