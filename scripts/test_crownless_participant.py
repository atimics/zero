"""Actor loss masks, review status, split leakage and actual gradient checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from crownless_v2 import Config, Crownless
from train_crownless_participant import checked_rows, check_split, loss

ROOT = Path(__file__).resolve().parents[1]


class ParticipantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = Tokenizer.from_file(str(ROOT / 'models/crownless-core-v2/tokenizer.json'))

    def row(self, text='I can ask the price.'):
        prefix = 'crownless-person-v1\nself:["A","smith"]\nturn:\n'
        target = json.dumps({'kind': 'speech', 'text': text}, separators=(',', ':'))
        p, t = self.tokenizer.encode(prefix).ids, self.tokenizer.encode(target).ids
        return {'prompt': {'format': 'crownless-person-v1', 'text': prefix, 'tokens': p},
                'target_text': target, 'tokens': p + t, 'labels': [-100] * (len(p)-1) + t + [0],
                'review_status': 'approved_compact', 'source_review_status': 'approved',
                'world_group': 'world-a'}

    def read(self, row, smoke=False):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rows.jsonl'
            path.write_text(json.dumps(row)+'\n')
            return checked_rows(path, self.tokenizer, smoke)

    def test_actor_target_round_trip(self):
        row = self.row()
        self.assertEqual(self.read(row), [row])

    def test_pending_rows_are_diagnostic_only(self):
        row = self.row()
        row['review_status'] = 'pending_compact_review'
        row['source_review_status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'approval'):
            self.read(row)
        self.assertEqual(self.read(row, smoke=True), [row])
        row['source_review_status'] = 'rejected_entity_spelling'
        with self.assertRaisesRegex(ValueError, 'excluded'):
            self.read(row, smoke=True)

    def test_prefix_or_eos_loss_corruption_is_rejected(self):
        for where in (0, -1):
            row = self.row()
            row['labels'][where] = 42
            with self.assertRaisesRegex(ValueError, 'loss mask'):
                self.read(row)

    def test_compiled_token_corruption_is_rejected(self):
        row = self.row()
        row['tokens'][0] = 42
        with self.assertRaisesRegex(ValueError, 'tokenizer differs'):
            self.read(row)

    def test_shared_world_and_content_are_rejected(self):
        first, second = self.row(), self.row('Let us find out which road is open today.')
        with self.assertRaisesRegex(ValueError, 'world history'):
            check_split([first], [second])
        first = self.row('Let us find out which road is open today.')
        second['world_group'] = 'world-b'
        with self.assertRaisesRegex(ValueError, 'target text'):
            check_split([first], [second])
        check_split([self.row()], [second])

    def test_multi_actor_output_is_rejected(self):
        row = self.row()
        row['target_text'] = json.dumps([{'kind': 'speech', 'text': 'A'}, {'kind': 'speech', 'text': 'B'}])
        with self.assertRaisesRegex(ValueError, 'one object'):
            self.read(row)

    def test_real_loss_uses_target_counts_and_updates_weights(self):
        torch.manual_seed(7)
        model = Crownless(Config(dim=16, heads=2, layers=1, ff=24, kinds=0), 'conversation')
        rows = [self.row(), self.row('Can we ask what the work pays and how long it lasts?')]
        counts = [sum(t != -100 for t in r['labels']) for r in rows]
        expected = sum(loss(model, [r], 'cpu') * n for r, n in zip(rows, counts)) / sum(counts)
        combined = loss(model, rows, 'cpu')
        self.assertAlmostEqual(combined.item(), expected.item(), places=5)
        before = model.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        optimizer.zero_grad()
        combined.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.embedding.weight))
        self.assertTrue(torch.isfinite(loss(model, rows, 'cpu')))


if __name__ == '__main__':
    unittest.main()
