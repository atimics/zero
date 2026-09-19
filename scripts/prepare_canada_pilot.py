"""Package the Canada corpus training pilot within a US$2 budget."""
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


def prepare_pilot(delivery, output, instance_type="g5.xlarge", region="ca-central-1"):
    if instance_type not in ["g5.xlarge", "g5.2xlarge"]:
        raise ValueError("Choose an A10G pilot host")
    limit = 75 if instance_type == "g5.2xlarge" else 85
    workload = (limit - 10) * 60
    inputs = delivery_files(delivery)
    manifest = prepare(output, subnet=None, instance_type=instance_type, region=region)
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
    script = script.replace(original_sha, digest(source)).replace('shutdown -h +30', f'shutdown -h +{limit}')
    script = script.replace('timeout 900 "$PYTHON" scripts/aws_canada_workload.py --output output/timing.json',
        f'timeout {workload} "$PYTHON" scripts/aws_canada_pilot.py --delivery delivery --data prepared --output output')
    (output / 'user-data.template.sh').write_text(script)
    stack = read_json(output / 'stack.json')
    code = stack['Resources']['Watchdog']['Properties']['Code']['ZipFile']
    stack['Resources']['Watchdog']['Properties']['Code']['ZipFile'] = code.replace('>= 1800', f'>= {limit * 60}')
    write_json(output / 'stack.json', stack)
    manifest.update(scope='Four Canada corpus pilot arms, seed 7, then paired held-out loss scoring',
        workload_timeout_seconds=workload, watchdog_maximum_age_seconds=limit * 60,
        result_file='pilot.json', controller_wait_seconds=(limit + 5) * 60,
        planning_instance_usd_at_watchdog_plus_one_minute=manifest["hourly_instance_usd"] * (limit + 1) / 60,
        shutdown=f'Guest shutdown at success/failure and +{limit} minutes; AWS watchdog at {limit} minutes',
        corpus_location=region, source_rights_evidence_scope='Canada', delivery_files={name: digest(path) for name, path in inputs.items()})
    manifest.pop('planning_instance_usd_at_31_minutes')
    manifest['files'] = {name: digest(output / name) for name in manifest['files']}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--instance-type', choices=['g5.xlarge', 'g5.2xlarge'], default='g5.xlarge')
    p.add_argument('--region', choices=['ca-central-1', 'us-west-2'], default='ca-central-1')
    a = p.parse_args(); prepare_pilot(a.delivery, a.output, a.instance_type, a.region)
