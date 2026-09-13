"""Build a reviewable, hash-bound Canada GPU timing launch package."""
import argparse
import datetime
import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

from canada_narrative import ROOT, EXPERIMENT, digest, read_json, write_json


def template(run_id, end_date):
    def trust(service):
        return {'Version': '2012-10-17', 'Statement': [
            {'Effect': 'Allow', 'Principal': {'Service': service}, 'Action': 'sts:AssumeRole'}]}
    def policy(statements):
        return [{'PolicyName': 'timing-only', 'PolicyDocument': {'Version': '2012-10-17', 'Statement': statements}}]
    def allow(actions, resource, **extra):
        return {'Effect': 'Allow', 'Action': actions, 'Resource': resource, **extra}
    resources = {
        'Bucket': {'Type': 'AWS::S3::Bucket', 'Properties': {
            'PublicAccessBlockConfiguration': {k: True for k in [
                'BlockPublicAcls', 'BlockPublicPolicy', 'IgnorePublicAcls', 'RestrictPublicBuckets']},
            'BucketEncryption': {'ServerSideEncryptionConfiguration': [
                {'ServerSideEncryptionByDefault': {'SSEAlgorithm': 'AES256'}}]},
            'LifecycleConfiguration': {'Rules': [{'Id': 'temporary-timing', 'Status': 'Enabled', 'ExpirationInDays': 1}]}}},
        'SecurityGroup': {'Type': 'AWS::EC2::SecurityGroup', 'Properties': {
            'VpcId': 'vpc-c64ccdaf', 'GroupDescription': 'ZERO timing outbound HTTPS',
            'SecurityGroupEgress': [{'IpProtocol': 'tcp', 'FromPort': 443, 'ToPort': 443, 'CidrIp': '0.0.0.0/0'}]}},
        'InstanceRole': {'Type': 'AWS::IAM::Role', 'Properties': {
            'AssumeRolePolicyDocument': trust('ec2.amazonaws.com'),
            'Policies': policy([
                allow('s3:GetObject', {'Fn::Sub': '${Bucket.Arn}/source.tar.gz'}),
                allow('s3:PutObject', {'Fn::Sub': '${Bucket.Arn}/results/*'}),
                allow('s3:ListBucket', {'Fn::GetAtt': ['Bucket', 'Arn']},
                      Condition={'StringLike': {'s3:prefix': ['results/*']}})])}},
        'Profile': {'Type': 'AWS::IAM::InstanceProfile', 'Properties': {'Roles': [{'Ref': 'InstanceRole'}]}},
        'WatchRole': {'Type': 'AWS::IAM::Role', 'Properties': {
            'AssumeRolePolicyDocument': trust('lambda.amazonaws.com'),
            'Policies': policy([
                allow('ec2:DescribeInstances', '*'),
                allow('ec2:TerminateInstances', {'Fn::Sub': 'arn:${AWS::Partition}:ec2:${AWS::Region}:${AWS::AccountId}:instance/*'},
                      Condition={'StringEquals': {'ec2:ResourceTag/ZeroTimingRun': run_id}})])}},
        'Watchdog': {'Type': 'AWS::Lambda::Function', 'Properties': {
            'Runtime': 'python3.12', 'Handler': 'index.handler', 'Timeout': 30, 'MemorySize': 128,
            'Role': {'Fn::GetAtt': ['WatchRole', 'Arn']},
            'Environment': {'Variables': {'RUN_ID': run_id}},
            'Code': {'ZipFile': (ROOT / 'scripts/aws_canada_watchdog.py').read_text()}}},
        'ScheduleRole': {'Type': 'AWS::IAM::Role', 'Properties': {
            'AssumeRolePolicyDocument': trust('scheduler.amazonaws.com'),
            'Policies': policy([allow('lambda:InvokeFunction', {'Fn::GetAtt': ['Watchdog', 'Arn']})])}},
        'Schedule': {'Type': 'AWS::Scheduler::Schedule', 'Properties': {
            'ScheduleExpression': 'rate(1 minute)', 'FlexibleTimeWindow': {'Mode': 'OFF'},
            'EndDate': end_date,
            'Target': {'Arn': {'Fn::GetAtt': ['Watchdog', 'Arn']},
                       'RoleArn': {'Fn::GetAtt': ['ScheduleRole', 'Arn']},
                       'RetryPolicy': {'MaximumRetryAttempts': 2, 'MaximumEventAgeInSeconds': 120}}}}
    }
    return {'AWSTemplateFormatVersion': '2010-09-09', 'Description': 'Temporary ZERO Canada GPU timing resources',
            'Resources': resources, 'Outputs': {name: {'Value': {'Ref': name}}
                                               for name in ['Bucket', 'SecurityGroup', 'Profile']}}


def package_source(output):
    lock = read_json(EXPERIMENT / 'implementation.lock.json')
    files = {name: ROOT / name for name in lock['files']}
    for name in ['experiments/canada-narrative-v1/implementation.lock.json', 'scripts/aws_canada_workload.py']:
        files[name] = ROOT / name
    hashes = {name: digest(file) for name, file in files.items()}
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode='wb', filename='', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            contents = {name: file.read_bytes() for name, file in files.items()}
            contents['SHA256SUMS'] = ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)).encode()
            for name, data in sorted(contents.items()):
                info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    (output / 'source.tar.gz').write_bytes(payload.getvalue())
    return hashes


def prepare(output, now=None, subnet="subnet-a24506fe", instance_type='g6.xlarge'):
    if subnet not in [None, "subnet-11e16978", "subnet-8f2684f4", "subnet-a24506fe"]:
        raise ValueError("Choose a verified Canada Central subnet")
    prices = {'g6.xlarge': .8936, 'g6.2xlarge': 1.08547}
    if instance_type not in prices:
        raise ValueError('Choose a single-L4 instance within the timing budget')
    price = prices[instance_type]
    now = int(time.time()) if now is None else now
    output.mkdir(parents=True, exist_ok=False)
    files = package_source(output)
    source_sha = digest(output / 'source.tar.gz')
    run_id = 'zero-canada-timing-' + source_sha[:12] + '-' + str(now)
    end = datetime.datetime.fromtimestamp(now + 90000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    write_json(output / 'stack.json', template(run_id, end))
    bootstrap = (ROOT / 'scripts/aws_canada_bootstrap.sh').read_text().replace('__SOURCE_SHA__', source_sha)
    (output / 'user-data.template.sh').write_text(bootstrap)
    request = {'ImageId': 'ami-092e8a40239c8733d', 'InstanceType': instance_type, 'MinCount': 1, 'MaxCount': 1,
               'ClientToken': hashlib.sha256(run_id.encode()).hexdigest(),
               'InstanceInitiatedShutdownBehavior': 'terminate',
               'MetadataOptions': {'HttpTokens': 'required', 'HttpEndpoint': 'enabled', 'HttpPutResponseHopLimit': 1},
               'NetworkInterfaces': [{'DeviceIndex': 0, 'SubnetId': subnet,
                                      'Groups': ['__STACK_SECURITY_GROUP__'], 'AssociatePublicIpAddress': True,
                                      'DeleteOnTermination': True}],
               'IamInstanceProfile': {'Name': '__STACK_PROFILE__'},
               'BlockDeviceMappings': [{'DeviceName': '/dev/sda1', 'Ebs': {
                   'VolumeSize': 50, 'VolumeType': 'gp3', 'Encrypted': True, 'DeleteOnTermination': True}}],
               'TagSpecifications': [{'ResourceType': resource, 'Tags': [
                   {'Key': 'Name', 'Value': run_id}, {'Key': 'ZeroTimingRun', 'Value': run_id},
                   {'Key': 'Project', 'Value': 'zero'}]} for resource in ['instance', 'volume']]}
    if subnet is None:
        del request['NetworkInterfaces']
        request['SecurityGroupIds'] = ['__STACK_SECURITY_GROUP__']
    write_json(output / 'request.template.json', request)
    (output / 'launch.py').write_bytes((ROOT / 'scripts/run_canada_aws.py').read_bytes())
    manifest = {'schema': 1, 'status': 'prepared-awaiting-paid-launch-approval', 'region': 'ca-central-1',
                'account': '022118847419', 'run_id': run_id, 'created_at': now, 'launch_valid_until': now + 86400,
                'instance_type': instance_type, 'instance_count': 1, 'hourly_instance_usd': price,
                'requested_budget_usd': 2, 'workload_timeout_seconds': 900,
                'watchdog_maximum_age_seconds': 1800, 'watchdog_check_interval_seconds': 60,
                'planning_instance_usd_at_31_minutes': price * 31 / 60,
                'scope': '105 synthetic updates, CUDA numerical check and timing; zero corpus training presentations',
                'shutdown': 'Guest shutdown at success/failure and +30 minutes; AWS watchdog checks instance age each minute',
                'cleanup': 'Collect results locally, confirm instance termination, empty temporary bucket and delete stack',
                'source_files': files,
                'files': {name: digest(output / name) for name in
                          ['source.tar.gz', 'stack.json', 'user-data.template.sh', 'request.template.json', 'launch.py']}}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instance-type', choices=['g6.xlarge', 'g6.2xlarge'], default='g6.xlarge')
    parser.add_argument('--automatic-zone', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, subnet=None if args.automatic_zone else 'subnet-a24506fe',
                             instance_type=args.instance_type), indent=2))
