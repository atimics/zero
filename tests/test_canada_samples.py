import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from canada_narrative import digest
from prepare_canada_samples import checkpoint_files


class SamplePackageTests(unittest.TestCase):
    def test_every_selected_checkpoint_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['AB-A', 'AB-B', 'BC-B', 'BC-C']:
                d = root / name; d.mkdir(); (d / 'best.pt').write_bytes(name.encode())
                (d / 'result.json').write_text(json.dumps({'best_sha256': digest(d / 'best.pt'),
                                                         'manifest_sha256': 'prepared'}))
            with patch('prepare_canada_samples.validate_pair') as validate:
                self.assertEqual(len(checkpoint_files(root)), 8)
                self.assertEqual(validate.call_count, 2)
                (root / 'BC-C/best.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'BC-C'):
                    checkpoint_files(root)


if __name__ == '__main__': unittest.main()
