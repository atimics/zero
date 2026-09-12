import json
from pathlib import Path
import tempfile
import unittest
import torch
from crownless_v2 import Crownless, Config, batch, encode_row, train_tokenizer


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_parameter_budget(self):
        self.assertEqual(sum(p.numel() for p in Crownless().parameters()), 4950337)

    def test_cache_and_causality(self):
        model = Crownless(Config(vocab=300, dim=24, layers=2, heads=3, ff=32, context=32)).eval()
        tokens = torch.randint(0, 300, (1, 12))
        meta = torch.zeros(1, 12, 5, dtype=torch.long)
        with torch.no_grad():
            whole, _ = model.hidden(tokens, meta)
            first, cache = model.hidden(tokens[:, :5], meta[:, :5])
            second, cache = model.hidden(tokens[:, 5:9], meta[:, 5:9], cache)
            third, _ = model.hidden(tokens[:, 9:], meta[:, 9:], cache)
            torch.testing.assert_close(torch.cat([first, second, third], 1), whole, atol=2e-6, rtol=2e-5)
            altered = tokens.clone()
            altered[:, 5:] = 17
            changed, _ = model.hidden(altered, meta)
            torch.testing.assert_close(changed[:, :5], whole[:, :5])

    def test_copy_labels_and_utf8(self):
        row = {'id': 'test', 'prefix': '- Éva met Mara.\n', 'output': 'Mara met Éva.',
               'fields': [{'field': 0, 'text': 'Éva', 'start': 2, 'end': 6, 'role': 1,
                           'knowledge': 0, 'provenance': 3, 'event': 1},
                          {'field': 1, 'text': 'Mara', 'start': 11, 'end': 15, 'role': 2,
                           'knowledge': 0, 'provenance': 3, 'event': 1}],
               'copies': [{'field': 1, 'text': 'Mara', 'start': 0, 'end': 4, 'role': 2},
                          {'field': 0, 'text': 'Éva', 'start': 9, 'end': 13, 'role': 1}]}
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = train_tokenizer([row], Path(temporary) / 'tokenizer.json')
            record = encode_row(tokenizer, row)
            self.assertEqual([x for x in record['copy_targets'] if x >= 0], [1, 0])
            model = Crownless(Config(dim=24, layers=2, heads=3, ff=32, context=128))
            loss = model.loss(batch([record], 'cpu'))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.copy_start.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.roles.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__': unittest.main()
