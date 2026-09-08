import math
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measure_sample_failures import repetition, measure
from subword_model import sample


class TinyTokenizer:
    def encode(self, prompt):
        return type('Encoding', (), {'ids': [0]})()
    def decode(self, ids):
        return str(ids)


class TinyModel(torch.nn.Module):
    context = 64
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(40, dtype=torch.float32) / 40)
    def forward(self, x):
        return self.logits.expand(1, x.shape[1], 40).clone()


class SampleFailures(unittest.TestCase):
    def test_loop_counts(self):
        r = repetition('French French French French')['2']
        self.assertEqual(r['max_count'], 3)
        self.assertEqual(r['max_frequency'], 1)
        self.assertAlmostEqual(r['repeat_excess_rate'], 2/3)
        self.assertEqual(repetition('one two three four')['2']['repeat_excess_rate'], 0)

    def test_prompt_names_excluded(self):
        result = measure([{'prompt': 'Sara ', 'output': 'Sara Tom met Huck.'}],
                         {'names': {'Sara': {'1': 5}, 'Tom': {'2': 5}, 'Huck': {'2': 9}}})
        self.assertEqual(result['rows'][0]['distinct_source_books'], 1)
        self.assertEqual(set(result['rows'][0]['matched_names']), {'Tom', 'Huck'})

    def test_probability_division_matches_penalty(self):
        model = TinyModel()
        for penalty in [1.0, 1.1]:
            expected = [0]
            rng = torch.Generator().manual_seed(7)
            for _ in range(70):
                weights = model.logits.detach().softmax(-1).clone()
                weights[list(set(expected[-64:]))] /= penalty
                weights = weights.pow(1/.7)
                values, indices = weights.topk(40)
                expected.append(indices[torch.multinomial(values / values.sum(), 1, generator=rng)].item())
            _, actual = sample(model, TinyTokenizer(), '', count=70, return_tokens=True, repetition_penalty=penalty)
            self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
