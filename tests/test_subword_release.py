import hashlib
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Release(unittest.TestCase):
    def test_weights_receipts(self):
        for name,item in json.loads((ROOT/'models/subword/manifest.json').read_text()).items():
            self.assertEqual(hashlib.sha256((ROOT/f'models/subword/{name}.pt').read_bytes()).hexdigest(),item['weights_sha256'])
    def test_sample_grid_and_raw_artifact(self):
        data=json.loads((ROOT/'docs/subword/results.json').read_text())
        self.assertIn('g irl',data['models']['5m-1024']['samples'][1]['output'])
        grid=json.loads((ROOT/'docs/subword/seed-grid.json').read_text())['samples']
        self.assertEqual(len(grid),96)
        self.assertEqual(len({(r['model'],r['prompt'],r['seed']) for r in grid}),96)
        for row in grid:self.assertTrue(row['output'].startswith(row['prompt']))
    def test_matched_evaluation(self):
        data=json.loads((ROOT/'docs/subword/results.json').read_text())
        for split in ['validation','test']:
            counts={data['baseline'][split]['target_bytes']}|{m['result'][split]['target_bytes'] for m in data['models'].values()}
            self.assertEqual(len(counts),1)
        a,b=[m['history'] for m in data['models'].values()]
        self.assertEqual([r['tokens'] for r in a],[r['tokens'] for r in b])
if __name__=='__main__':unittest.main()
