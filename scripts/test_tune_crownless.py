"""Check audit alignment and the output-only training boundary."""
import json
import tempfile
import unittest
from pathlib import Path

from tune_crownless import batch, read_split


class CorpusTests(unittest.TestCase):
    def fixture(self, root, prefix=b'- old\n- new\n', target=b'News.\n'):
        data = prefix + target + b'\n'
        (root / 'train.txt').write_bytes(data)
        row = dict(id='a', input=dict(kind='NOTICE'), output=target.decode().rstrip('\n'),
                   text_start=0, output_start=len(prefix), text_bytes=len(data))
        (root / 'train.audit.jsonl').write_text(json.dumps(row) + '\n')

    def test_target_boundary_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            examples = read_split(root, 'train', 512)
            examples.append(dict(prefix=b'- x\n', target=b'Y\n'))
            x, y = batch(examples, 'cpu')
            self.assertEqual(y[0, :11].tolist(), [-100] * 11)
            self.assertEqual(y[0, 11:].tolist(), list(b'News.\n'))
            self.assertEqual(x[0, 11].item(), 10)
            self.assertEqual(y[1, 3:5].tolist(), list(b'Y\n'))
            self.assertTrue((y[1, 5:] == -100).all())

    def test_trim_keeps_final_event_and_entire_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            e = read_split(root, 'train', 11)[0]
            self.assertEqual(e['prefix'], b'- new\n')
            self.assertEqual(e['target'], b'News.\n')
            self.assertEqual(e['dropped'], 1)
            with self.assertRaises(ValueError):
                read_split(root, 'train', 5)

    def test_bad_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'train.txt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                read_split(root, 'train', 512)


if __name__ == '__main__':
    unittest.main()
