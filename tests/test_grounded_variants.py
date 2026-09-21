import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from grounded_dialogue import build, exchange_rows, variants
from run_reviewed_dialogue_pilot import read, encode
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'experiments/grounded-dialogue/input'


class GroundedVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read(ROOT / 'experiments/full-event-rehearsal/input/rehearsal.jsonl')
        cls.bank = json.loads((INPUT / 'dialogue-bank-v2.json').read_text())['exchanges']
        cls.original = json.loads((INPUT / 'dialogue-bank.json').read_text())['exchanges']
        cls.meta = {'meaning_ids': {k: i + 1 for i, k in enumerate(cls.bank)}}
        cls.tok = Tokenizer.from_file(str(ROOT / 'models/crownless-core-v2/tokenizer.json'))

    def test_every_meaning_has_the_original_exchange(self):
        self.assertEqual(set(self.bank), set(self.original))
        for rule, entry in self.bank.items():
            self.assertEqual(variants(entry)[0], self.original[rule])

    def test_every_variant_encodes_with_copy_targets(self):
        selected = {}
        for row in self.rows:
            selected.setdefault(row['rule'], row)
        result = build(list(selected.values()), self.bank, self.meta)
        self.assertEqual(len(result), sum(len(variants(v)) for v in self.bank.values()) * 4)
        for row in result:
            encoded = encode(self.tok, row)
            self.assertEqual(sum(c >= 0 for c in encoded['copy_targets']), len(row['copies']))

    def test_alternates_target_a_different_field(self):
        import re
        rows = {}
        for line in (ROOT / 'experiments/full-event-rehearsal/input/grammar-test.jsonl').read_text().splitlines():
            row = json.loads(line)
            rows.setdefault(row['rule'], row)
        for rule, entry in self.bank.items():
            for exchange in variants(entry)[1:]:
                before = [int(x) for x in re.findall(r'\{(\d+)\}', self.original[rule][1])]
                after = [int(x) for x in re.findall(r'\{(\d+)\}', exchange[1])]
                self.assertTrue(after, rule)
                self.assertNotEqual(set(after), set(before), rule)

    def test_training_wording_is_disjoint_from_the_contrast_probes(self):
        contrasts = json.loads((INPUT / 'question-contrasts.json').read_text())['questions']
        for rule, probe in contrasts.items():
            for exchange in variants(self.bank[rule]):
                self.assertNotEqual(exchange[0], probe[0], rule)


if __name__ == '__main__':
    unittest.main()