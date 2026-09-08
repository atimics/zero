import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from zero_torch import load, update, write


class TorchParity(unittest.TestCase):
    def test_native_forward_gradient_update_and_roundtrip(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "reference"
            subprocess.run(["cc", "-O1", "-std=c11", str(ROOT / "tests/torch_reference.c"),
                            "-o", str(binary), "-lm"], check=True)
            subprocess.run([str(binary), "parity", "initial.ckpt"], cwd=root, check=True)
            model, raw_moments = load(root / "initial.ckpt")
            moments = [[torch.from_numpy(a).reshape(w.shape) for a in pair]
                       for w, pair in zip(model.weights, raw_moments)]
            tokens = torch.arange(32, 48).reshape(1, -1)
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), (tokens + 1).flatten())
            loss.backward()
            reference = np.fromfile(root / "reference.bin", dtype=np.float32)
            probabilities = logits.softmax(-1).detach().numpy().flatten()
            np.testing.assert_allclose(probabilities, reference[:2048], atol=2e-7, rtol=2e-5)
            gradients = np.concatenate([w.grad.numpy().flatten() for w in model.weights])
            np.testing.assert_allclose(gradients, reference[2048:], atol=2e-6, rtol=2e-3)
            update(model, moments, 1, 0.0003)
            native, _ = load(root / "updated.ckpt")
            for got, expected in zip(model.weights, native.weights):
                np.testing.assert_allclose(got.detach().numpy(), expected.detach().numpy(), atol=3e-6, rtol=2e-4)
            write(root / "roundtrip.ckpt", model, moments, 1)
            roundtrip, _ = load(root / "roundtrip.ckpt")
            self.assertTrue(torch.equal(model(tokens), roundtrip(tokens)))


if __name__ == "__main__":
    unittest.main()
