"""Launch or recover the approved, bounded Canada GPU timing package."""
import argparse
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify(directory, expected):
    if sha(directory / 'manifest.json') != expected:
        raise ValueError('Use the approved manifest hash')
    manifest = read(directory / 'manifest.json')
    if manifest['region'] != 'ca-central-1' or manifest['instance_count'] != 1:
        raise ValueError('Launch scope differs from one Canada instance')
    for name, expected_sha in manifest['files'].items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory.resolve()) or sha(file) != expected_sha:
            raise ValueError('Package file identity mismatch: ' + name)
    if sha(Path(__file__)) != manifest['files']['launch.py']:
        raise ValueError('Launcher identity mismatch')
    return manifest


def run(directory, expected, mode):
    manifest = verify(directory, expected)
    region = manifest['region']; run_id = manifest['run_id']
    def aws(*args, raw=False, timeout=120):
        completed = subprocess.run(['aws', '--region', region, '--no-cli-pager', *args],
                                   text=True, capture_output=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError('AWS ' + args[0] + ' failed: ' + completed.stderr[-1500:])
        return completed.stdout if raw or not completed.stdout.strip() else json.loads(completed.stdout)
    if aws('sts', 'get-caller-identity')['Account'] != manifest['account']:
        raise ValueError('AWS account differs from the approved package')
    state_path = directory / 'launch-state.json'
    if mode == 'launch':
        if time.time() >= manifest['launch_valid_until']:
            raise ValueError('Prepare fresh launch settings for this timing package')
        if state_path.exists():
            raise ValueError('Use collect to recover this existing launch')
        write(state_path, {'run_id': run_id, 'stage': 'creating-stack'})
        aws('cloudformation', 'create-stack', '--stack-name', run_id,
            '--template-body', 'file://' + str(directory / 'stack.json'),
            '--capabilities', 'CAPABILITY_IAM', '--tags', 'Key=ZeroTimingRun,Value=' + run_id)
        aws('cloudformation', 'wait', 'stack-create-complete', '--stack-name', run_id, raw=True, timeout=600)
    stack = aws('cloudformation', 'describe-stacks', '--stack-name', run_id)['Stacks'][0]
    if {'Key': 'ZeroTimingRun', 'Value': run_id} not in stack.get('Tags', []):
        raise ValueError('Stack ownership differs from this timing run')
    outputs = {r['OutputKey']: r['OutputValue'] for r in stack.get('Outputs', [])}
    if not outputs and mode == 'collect' and stack['StackStatus'] in ['ROLLBACK_COMPLETE', 'CREATE_FAILED']:
        aws('cloudformation', 'delete-stack', '--stack-name', run_id)
        aws('cloudformation', 'wait', 'stack-delete-complete', '--stack-name', run_id, raw=True, timeout=600)
        write(state_path, {'run_id': run_id, 'stage': 'cleaned-after-stack-failure'})
        raise RuntimeError('Stack creation failed; temporary resources were cleaned up')
    if not outputs:
        raise RuntimeError('Stack setup is still in progress; use collect after it completes')
    bucket = outputs['Bucket']

    def instances():
        response = aws('ec2', 'describe-instances', '--filters', 'Name=tag:ZeroTimingRun,Values=' + run_id)
        return [i for r in response['Reservations'] for i in r['Instances']]

    try:
        if mode == 'launch':
            aws('s3', 'cp', str(directory / 'source.tar.gz'), f's3://{bucket}/source.tar.gz', '--only-show-errors', raw=True)
            request = read(directory / 'request.template.json')
            request['NetworkInterfaces'][0]['Groups'] = [outputs['SecurityGroup']]
            request['IamInstanceProfile']['Name'] = outputs['Profile']
            script = (directory / 'user-data.template.sh').read_text().replace('__BUCKET__', bucket)
            request['UserData'] = base64.b64encode(script.encode()).decode()
            write(directory / 'request.json', request)
            receipt = aws('ec2', 'run-instances', '--cli-input-json', 'file://' + str(directory / 'request.json'))
            write(directory / 'instance.json', receipt)
            write(state_path, {'run_id': run_id, 'stage': 'timing', 'bucket': bucket,
                               'instance_id': receipt['Instances'][0]['InstanceId']})
        # A local timeout supplements the independent AWS watchdog.
        deadline = time.monotonic() + 2100
        while time.monotonic() < deadline:
            active = [i for i in instances() if i['State']['Name'] != 'terminated']
            if not active:
                break
            print('Timing instance: ' + active[0]['State']['Name'], flush=True)
            time.sleep(15)
    finally:
        active = [i['InstanceId'] for i in instances() if i['State']['Name'] != 'terminated']
        if active:
            aws('ec2', 'terminate-instances', '--instance-ids', *active)
            aws('ec2', 'wait', 'instance-terminated', '--instance-ids', *active, raw=True, timeout=600)
        results = directory / 'results'; results.mkdir(exist_ok=True)
        aws('s3', 'sync', f's3://{bucket}/results/', str(results), '--only-show-errors', raw=True)
        # Persist results locally before removing this run's temporary resources.
        write(directory / 'collection.json', {'run_id': run_id,
              'instances': [{'id': i['InstanceId'], 'state': i['State']['Name']} for i in instances()],
              'files': {p.name: sha(p) for p in results.iterdir() if p.is_file()}})
        aws('s3', 'rm', f's3://{bucket}', '--recursive', '--only-show-errors', raw=True)
        aws('cloudformation', 'delete-stack', '--stack-name', run_id)
        aws('cloudformation', 'wait', 'stack-delete-complete', '--stack-name', run_id, raw=True, timeout=600)
        write(state_path, {'run_id': run_id, 'stage': 'cleaned', 'instance_terminated': True})
    timing = results / 'timing.json'; finish = results / 'finish.json'
    if not timing.exists() or not finish.exists() or read(finish)['exit_code'] != 0 or read(timing)['status'] != 'passed':
        raise RuntimeError('Timing failed; inspect the collected bootstrap log')
    print(json.dumps(read(timing), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['launch', 'collect'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--approved-manifest-sha256', required=True)
    args = parser.parse_args()
    run(args.directory.resolve(), args.approved_manifest_sha256, args.mode)
