import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from crownless_v2 import Crownless, Config, batch, encode_row, generate, train_tokenizer
from crownless_v2_export import export, load_export


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
            altered = copy.deepcopy(row)
            replacement = 'A Very Long Name'
            delta = len(replacement.encode()) - len('Éva'.encode())
            altered['prefix'] = row['prefix'].replace('Éva', replacement)
            altered['output'] = row['output'].replace('Éva', replacement)
            altered['fields'][0].update(text=replacement, end=6 + delta)
            for key in ('start', 'end'): altered['fields'][1][key] += delta
            altered['copies'][1].update(text=replacement, end=13 + delta)
            a = encode_row(tokenizer, row, slots=True)
            b = encode_row(tokenizer, altered, slots=True)
            self.assertEqual(a['tokens'], b['tokens'])
            self.assertEqual(a['meta'], b['meta'])
            self.assertEqual(a['candidates'], b['candidates'])
            model.mode = 'slots'
            with torch.no_grad():
                model.copy_gate.weight.zero_()
                model.copy_gate.bias.fill_(10)
                model.copy_start.weight.zero_()
                model.copy_end.weight.zero_()
            one = generate(model, tokenizer, a, max_tokens=2)
            two = generate(model, tokenizer, b, max_tokens=2)
            self.assertEqual(one['actions'], two['actions'])
            self.assertEqual(one['text'], 'ÉvaÉva')
            self.assertEqual(two['text'], replacement * 2)
            reordered = copy.deepcopy(row)
            reordered['prefix'] = '- Mara was met by Éva.\n'
            for field in reordered['fields']:
                start = reordered['prefix'].index(field['text'])
                field['start'] = len(reordered['prefix'][:start].encode())
                field['end'] = field['start'] + len(field['text'].encode())
            packet = encode_row(tokenizer, row, slots=True, packet=True)
            paraphrase = encode_row(tokenizer, reordered, slots=True, packet=True)
            self.assertEqual(packet['tokens'], paraphrase['tokens'])
            self.assertEqual(packet['meta'], paraphrase['meta'])
            exported = Path(temporary) / 'core.ccv2'
            token_path = Path(temporary) / 'tokenizer.json'
            export(model, token_path, exported)
            quantized, _ = load_export(exported, token_path)
            self.assertEqual(generate(quantized, tokenizer, a, max_tokens=2), one)
            damaged = bytearray(exported.read_bytes())
            damaged[-1] ^= 1
            exported.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, 'payload'): load_export(exported, token_path)


if __name__ == '__main__': unittest.main()
