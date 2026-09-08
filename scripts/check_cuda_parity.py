"""Check the actual CUDA device against the frozen native full-model loss."""
import json
from pathlib import Path
import numpy as np
import torch
from zero_torch import load

torch.backends.cuda.matmul.allow_tf32 = False
model, _ = load("initial.ckpt")
model = model.cuda()
tokens = torch.from_numpy(np.frombuffer(Path("data/validation.txt").read_bytes()[:513], dtype=np.uint8).copy()).long().cuda()[None, :]
with torch.no_grad():
    loss = torch.nn.functional.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten()).item()
reference = json.loads(Path("parity.json").read_text())["native_loss"]
if abs(loss - reference) > 0.0001:
    raise RuntimeError(f"CUDA/native loss difference: {loss} versus {reference}")
print(json.dumps({"cuda_loss": loss, "native_loss": reference, "difference": abs(loss-reference)}), flush=True)
