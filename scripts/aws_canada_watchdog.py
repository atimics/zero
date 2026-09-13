"""Terminate this timing run after thirty minutes, including failed boot."""
import datetime
import os


def expired(instance, now):
    return (now - instance['LaunchTime']).total_seconds() >= 1800


def handler(event, context):
    import boto3
    ec2 = boto3.client('ec2')
    now = datetime.datetime.now(datetime.timezone.utc)
    ids = []
    for page in ec2.get_paginator('describe_instances').paginate(Filters=[
        {'Name': 'tag:ZeroTimingRun', 'Values': [os.environ['RUN_ID']]},
        {'Name': 'instance-state-name', 'Values': ['pending', 'running', 'stopping', 'stopped']}
    ]):
        ids.extend(i['InstanceId'] for r in page['Reservations'] for i in r['Instances'] if expired(i, now))
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    return {'terminated': ids}
