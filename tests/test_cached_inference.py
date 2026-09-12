import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from subword_model import create, sample

class Tokens:
    def encode(self, text):
        return type('Encoded', (), {'ids': [int(x) for x in text.split()]})()
    def decode(self, ids):
        return ' '.join(map(str, ids))

class CachedInference(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = create(dict(vocab=48, context=16, dim=32, heads=4, layers=2, ff=64)).eval()

    def test_prefix_and_chunk_logits(self):
        ids = torch.tensor([[2, 8, 3, 19, 25, 4, 7]])
        logits, cache = self.model.forward_cached(ids[:, :3])
        torch.testing.assert_close(logits, self.model(ids[:, :3])[:, -1:], atol=2e-6, rtol=2e-5)
        logits, cache = self.model.forward_cached(ids[:, 3:6], cache)
        torch.testing.assert_close(logits, self.model(ids[:, :6])[:, -1:], atol=2e-6, rtol=2e-5)
        logits, cache = self.model.forward_cached(ids[:, 6:], cache)
        torch.testing.assert_close(logits, self.model(ids)[:, -1:], atol=2e-6, rtol=2e-5)

    def test_sampling_past_context_rebuild(self):
        for prompt in ['1 2 3', ' '.join(str(i) for i in range(20))]:
            plain = sample(self.model, Tokens(), prompt, count=32, return_tokens=True)
            cached = sample(self.model, Tokens(), prompt, count=32, return_tokens=True, use_cache=True)
            self.assertEqual(plain, cached)

    def test_request_cache_is_separate(self):
        prompt = '2 3 4'
        first = sample(self.model, Tokens(), prompt, count=20, use_cache=True)
        sample(self.model, Tokens(), '5 6 7', count=20, use_cache=True)
        self.assertEqual(first, sample(self.model, Tokens(), prompt, count=20, use_cache=True))

if __name__ == '__main__':
    unittest.main()
