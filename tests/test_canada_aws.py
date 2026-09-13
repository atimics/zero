import datetime
import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from aws_canada_watchdog import expired
from prepare_canada_aws import prepare
from run_canada_aws import verify, sha, run


class CanadaAwsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name) / 'package'
        self.manifest = prepare(self.directory, now=1789260000)

    def tearDown(self):
        self.temporary.cleanup()

    def test_package_reproduces(self):
        other = Path(self.temporary.name) / 'repeat'
        self.assertEqual(self.manifest, prepare(other, now=1789260000))
        self.assertEqual(sha(self.directory / 'source.tar.gz'), sha(other / 'source.tar.gz'))

    def test_source_contains_code_and_no_corpus(self):
        with tarfile.open(self.directory / 'source.tar.gz') as archive:
            names = archive.getnames()
            self.assertIn('scripts/aws_canada_workload.py', names)
            self.assertIn('SHA256SUMS', names)
            self.assertFalse(any(name.endswith(('.bin', '.pt', '.jsonl')) for name in names))
            self.assertTrue(all(name.startswith(('scripts/', 'experiments/')) or name == 'SHA256SUMS' for name in names))

    def test_approved_hash_and_mutation(self):
        expected = sha(self.directory / 'manifest.json')
        self.assertEqual(verify(self.directory, expected)['instance_count'], 1)
        (self.directory / 'source.tar.gz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify(self.directory, expected)

    def test_request_limits_and_cleanup(self):
        request = json.loads((self.directory / 'request.template.json').read_text())
        self.assertEqual((request['MinCount'], request['MaxCount']), (1, 1))
        self.assertEqual(request['InstanceType'], 'g6.xlarge')
        self.assertEqual(request['InstanceInitiatedShutdownBehavior'], 'terminate')
        self.assertEqual(request['MetadataOptions']['HttpTokens'], 'required')
        self.assertTrue(request['BlockDeviceMappings'][0]['Ebs']['DeleteOnTermination'])
        self.assertTrue(request['BlockDeviceMappings'][0]['Ebs']['Encrypted'])
        self.assertEqual(self.manifest['requested_budget_usd'], 2)

    def test_watchdog_deadline_including_stopped(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.assertFalse(expired({'LaunchTime': now - datetime.timedelta(seconds=1799)}, now))
        self.assertTrue(expired({'LaunchTime': now - datetime.timedelta(seconds=1800)}, now))
        stack = json.loads((self.directory / 'stack.json').read_text())['Resources']
        self.assertEqual(stack['Schedule']['Properties']['ScheduleExpression'], 'rate(1 minute)')
        self.assertRegex(stack['Schedule']['Properties']['EndDate'], r'\.000Z$')
        self.assertNotIn('ActionAfterCompletion', stack['Schedule']['Properties'])
        self.assertIn("'stopped'", stack['Watchdog']['Properties']['Code']['ZipFile'])
        statements = stack['WatchRole']['Properties']['Policies'][0]['PolicyDocument']['Statement']
        terminate = next(s for s in statements if s['Action'] == 'ec2:TerminateInstances')
        self.assertEqual(terminate['Condition']['StringEquals']['ec2:ResourceTag/ZeroTimingRun'], self.manifest['run_id'])

    def test_bootstrap_preserves_failure_receipt(self):
        script = (self.directory / 'user-data.template.sh').read_text()
        self.assertIn('trap finish EXIT', script)
        self.assertIn('shutdown -h +30', script)
        self.assertIn('timeout 900', script)
        self.assertIn('output/finish.json', script)
        self.assertLess(script.index('trap finish EXIT'), script.index('aws s3 cp'))

    def fake_aws(self, calls, fail_launch=False, fail_download=False):
        active = [False]
        def execute(argv, **kwargs):
            args = argv[6:]; service, operation = args[:2]; calls.append((service, operation))
            response = {}
            if service == 'sts':
                response = {'Account': self.manifest['account']}
            elif service == 'cloudformation' and operation == 'describe-stacks':
                response = {'Stacks': [{'Tags': [{'Key': 'ZeroTimingRun', 'Value': self.manifest['run_id']}],
                    'Outputs': [{'OutputKey': k, 'OutputValue': v} for k, v in
                                [('Bucket', 'private-test-bucket'), ('SecurityGroup', 'sg-test'), ('Profile', 'profile')]]}]}
            elif service == 'ec2' and operation == 'run-instances':
                active[0] = fail_launch
                if fail_launch:
                    return SimpleNamespace(returncode=1, stdout='', stderr='uncertain launch response')
                response = {'Instances': [{'InstanceId': 'i-test'}]}
            elif service == 'ec2' and operation == 'describe-instances':
                response = {'Reservations': [{'Instances': [{'InstanceId': 'i-test',
                             'State': {'Name': 'running' if active[0] else 'terminated'}}]}]}
            elif service == 'ec2' and operation == 'terminate-instances':
                active[0] = False
            elif service == 's3' and operation == 'sync':
                if fail_download:
                    return SimpleNamespace(returncode=1, stdout='', stderr='download failed')
                results = self.directory / 'results'; results.mkdir(exist_ok=True)
                (results / 'timing.json').write_text('{"status":"passed"}')
                (results / 'finish.json').write_text('{"exit_code":0}')
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr='')
        return execute

    def test_launch_collects_before_cleanup(self):
        calls = []
        with patch('run_canada_aws.subprocess.run', self.fake_aws(calls)), patch('run_canada_aws.time.time', return_value=1789260100):
            run(self.directory, sha(self.directory / 'manifest.json'), 'launch')
        self.assertLess(calls.index(('s3', 'sync')), calls.index(('s3', 'rm')))
        self.assertLess(calls.index(('s3', 'rm')), calls.index(('cloudformation', 'delete-stack')))
        self.assertEqual(json.loads((self.directory / 'launch-state.json').read_text())['stage'], 'cleaned')

    def test_uncertain_launch_terminates_tagged_instance(self):
        calls = []
        with patch('run_canada_aws.subprocess.run', self.fake_aws(calls, fail_launch=True)), patch('run_canada_aws.time.time', return_value=1789260100):
            with self.assertRaises(RuntimeError):
                run(self.directory, sha(self.directory / 'manifest.json'), 'launch')
        self.assertIn(('ec2', 'terminate-instances'), calls)
        self.assertIn(('cloudformation', 'delete-stack'), calls)

    def test_failed_collection_preserves_results_for_recovery(self):
        calls = []
        with patch('run_canada_aws.subprocess.run', self.fake_aws(calls, fail_download=True)), patch('run_canada_aws.time.time', return_value=1789260100):
            with self.assertRaises(RuntimeError):
                run(self.directory, sha(self.directory / 'manifest.json'), 'launch')
        self.assertNotIn(('s3', 'rm'), calls)
        self.assertNotIn(('cloudformation', 'delete-stack'), calls)


if __name__ == '__main__':
    unittest.main()
