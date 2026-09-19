import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_canada_pilot import prepare_pilot
from canada_narrative import read_json, digest
from run_canada_aws import verify


class PilotPackageTests(unittest.TestCase):
    def test_canada_scope_deadlines_results_and_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); data = root / 'input.txt'; data.write_text('fixture')
            package = root / 'package'
            with patch('prepare_canada_pilot.delivery_files', return_value={'input.txt': data}):
                manifest = prepare_pilot(root, package)
            self.assertEqual(verify(package, digest(package / 'manifest.json'))['region'], 'ca-central-1')
            self.assertEqual(manifest['requested_budget_usd'], 2)
            self.assertLess(manifest['planning_instance_usd_at_watchdog_plus_one_minute'], 1.7)
            self.assertEqual(manifest['result_file'], 'pilot.json')
            self.assertEqual(manifest['watchdog_maximum_age_seconds'], 5100)
            script = (package / 'user-data.template.sh').read_text()
            self.assertIn('shutdown -h +85', script)
            self.assertIn('timeout 4500', script)
            self.assertIn(digest(package / 'source.tar.gz'), script)
            self.assertIn('AWS_DEFAULT_REGION=ca-central-1', script)
            stack = read_json(package / 'stack.json')
            self.assertIn('>= 5100', stack['Resources']['Watchdog']['Properties']['Code']['ZipFile'])
            with tarfile.open(package / 'source.tar.gz') as archive:
                self.assertEqual(archive.extractfile('delivery/input.txt').read(), b'fixture')
                self.assertIn('scripts/aws_canada_pilot.py', archive.getnames())
            self.assertEqual(manifest['files']['launch.py'], digest(package / 'launch.py'))
            with patch('prepare_canada_pilot.delivery_files', return_value={'input.txt': data}):
                fallback = prepare_pilot(root, root / 'larger', 'g5.2xlarge')
            self.assertEqual(fallback['watchdog_maximum_age_seconds'], 4500)
            self.assertEqual(fallback['workload_timeout_seconds'], 3900)
            self.assertLess(fallback['planning_instance_usd_at_watchdog_plus_one_minute'], 1.71)


if __name__ == '__main__': unittest.main()
