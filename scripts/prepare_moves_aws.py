"""Prepare the content-addressed AWS package and a three-hour EC2 request.

The move-axis run is small -- a 5M model, 8000 steps -- so the deadline is
three hours rather than twelve, and it exists to bound a hang, not to fit the
work. The package pins every file it carries by hash, and the gate the run is
held to is fixed in the trainer's manifest before the instance starts.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# crownless_conversation.py is not imported by the trainer, but the run manifest
# hashes it for provenance, so the package carries it too.
SCRIPTS = ['scripts/crownless_v2.py', 'scripts/crownless_v2_export.py',
           'scripts/crownless_moves.py', 'scripts/crownless_conversation.py',
           'scripts/train_crownless_moves.py']
CORPUS = ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'wording.jsonl',
          'rules.json', 'manifest.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--chat', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', default='crownless.moves.v1')
    parser.add_argument('--hours', type=float, default=3.0)
    parser.add_argument('--instance-type', default='g5.xlarge')
    parser.add_argument('--hourly-usd', type=float, default=1.006)
    parser.add_argument('--steps', type=int, default=8000,
                        help='8000 steps at batch 16 is 1.3 passes over a 100k-row corpus')
    parser.add_argument('--guard-every', type=int, default=500,
                        help='Guard evaluation runs on CPU, so a longer run wants it less often')
    parser.add_argument('--extra', default='',
                        help='Further trainer flags, recorded in the manifest so the run says '
                             'what it was asked to do')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    files = {name: ROOT / name for name in SCRIPTS}
    files.update({f'corpus/{name}': args.corpus / name for name in CORPUS})
    files['base/core.ccv2'] = args.base
    files['base/tokenizer.json'] = args.tokenizer
    files['chat-rows-stance.jsonl'] = args.chat
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing: parser.error(f'Missing from the package: {missing}')

    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    checksum = ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)).encode()
    package = args.output / 'source.tar.gz'
    with package.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name in sorted(files):
                info = archive.gettarinfo(str(files[name]), arcname=name)
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ''
                with files[name].open('rb') as source:
                    archive.addfile(info, source)
            info = tarfile.TarInfo('SHA256SUMS')
            info.size = len(checksum)
            archive.addfile(info, io.BytesIO(checksum))

    sha = hashlib.sha256(package.read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip()
    seconds = int(args.hours * 3600)
    deadline = int(time.time()) + seconds
    prefix = f'experiments/crownless-moves-v1/cuda-{sha[:12]}-{deadline}'
    key = f'experiments/crownless-moves-v1/packages/{sha}.tar.gz'
    script = (ROOT / 'scripts/aws_moves_user_data.sh').read_text()
    for name, value in {'PREFIX': prefix, 'SOURCE_KEY': key, 'SOURCE_SHA256': sha,
                        'DEADLINE': str(deadline), 'EXPERIMENT': args.experiment,
                        'COMMIT': commit, 'STEPS': str(args.steps),
                        'GUARD_EVERY': str(args.guard_every), 'EXTRA': args.extra}.items():
        script = script.replace(f'__{name}__', value)
    if '__' in script.replace('__pycache__', ''): raise ValueError('Unfilled placeholder')

    request = {'ImageId': 'ami-0eb7d782cce2fe526', 'InstanceType': args.instance_type,
        'MinCount': 1, 'MaxCount': 1,
        'IamInstanceProfile': {'Name': 'zero-training-ec2'},
        'NetworkInterfaces': [{'DeviceIndex': 0, 'SubnetId': 'subnet-eb3e2f8e',
                               'Groups': ['sg-0059d0413ff74df6e'],
                               'AssociatePublicIpAddress': True, 'DeleteOnTermination': True}],
        'MetadataOptions': {'HttpTokens': 'required', 'HttpEndpoint': 'enabled'},
        'BlockDeviceMappings': [{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': 75, 'VolumeType': 'gp3',
                                  'Encrypted': True, 'DeleteOnTermination': True}}],
        'InstanceInitiatedShutdownBehavior': 'terminate',
        'ClientToken': f'crownless-moves-{sha[:24]}-{deadline}',
        'UserData': base64.b64encode(script.encode()).decode(),
        'TagSpecifications': [{'ResourceType': resource, 'Tags': [
              {'Key': 'Project', 'Value': 'crownless'},
              {'Key': 'Name', 'Value': 'crownless-moves'},
              {'Key': 'Experiment', 'Value': args.experiment},
              {'Key': 'SourceSha256', 'Value': sha},
              {'Key': 'SourceCommit', 'Value': commit},
              {'Key': 'Deadline', 'Value': str(deadline)}]} for resource in ['instance', 'volume']]}
    request_text = json.dumps(request, indent=2) + '\n'
    (args.output / 'request.json').write_text(request_text)
    (args.output / 'user-data.sh').write_text(script)
    manifest = {'experiment': args.experiment, 'package_sha256': sha,
                'package_bytes': package.stat().st_size, 'files': hashes,
                'steps': args.steps, 'guard_every': args.guard_every, 'extra': args.extra,
                'source_commit': commit, 'bucket': 'zero-training-022118847419',
                'source_key': key, 'result_prefix': prefix, 'deadline': deadline,
                'maximum_instance_seconds': seconds, 'instance_type': args.instance_type,
                'hourly_ec2_usd': args.hourly_usd,
                'deadline_ec2_usd': round(args.hourly_usd * args.hours, 2),
                'root_volume_gib': 75,
                'user_data_sha256': hashlib.sha256(script.encode()).hexdigest(),
                'request_sha256': hashlib.sha256(request_text.encode()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
