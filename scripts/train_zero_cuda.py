"""Bounded CUDA run of the ZERO literary model using native C initialization."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from zero_torch import load, update, write


@torch.no_grad()
def evaluate(model, data, sequences):
    model.eval()
    starts = torch.arange(sequences, device=data.device) * (len(data) - model.context - 1) // max(1, sequences - 1)
    losses = []
    for group in starts.split(8):
        window = data[group[:, None] + torch.arange(model.context + 1, device=data.device)].long()
        losses.append(F.cross_entropy(model(window[:, :-1]).flatten(0, 1), window[:, 1:].flatten()).item() * len(group))
    model.train()
    return sum(losses) / sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    if args.output.exists():
        raise SystemExit("Use a fresh output directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, raw_moments = load(args.initial)
    if sum(w.numel() for w in model.weights) != 4852992 or model.header[10] != 0:
        raise ValueError("Expected the fresh 4,852,992-parameter C initialization")
    model = model.cuda()
    moments = [[torch.from_numpy(a).reshape(w.shape).cuda() for a in pair]
               for w, pair in zip(model.weights, raw_moments)]
    data = {split: torch.tensor(np.frombuffer((args.data / f"{split}.txt").read_bytes(), dtype=np.uint8).copy(),
                                device="cuda") for split in ["train", "validation"]}
    if any(int(value.max()) >= model.vocab for value in data.values()):
        raise ValueError("Corpus contains a token outside the vocabulary")
    generator = torch.Generator(device="cuda").manual_seed(7)
    offsets = torch.arange(513, device="cuda")
    started = time.monotonic()
    best = float("inf")
    history = []
    training_loss = torch.zeros((), device="cuda")
    for step in range(1, args.steps + 1):
        if time.time() >= args.deadline - 120:
            raise TimeoutError("Saving time reserved before the instance deadline")
        starts = torch.randint(len(data["train"]) - 512, (2,), generator=generator, device="cuda")
        window = data["train"][starts[:, None] + offsets].long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(window[:, :-1], dropout=0.1).flatten(0, 1), window[:, 1:].flatten())
        loss.backward()
        lr = 0.0003 * min(1, step / 1000)
        if step > 1000 and args.steps > 1000:
            lr *= 0.5 * (1 + math.cos(math.pi * (step - 1000) / (args.steps - 1000)))
        update(model, moments, step, lr)
        training_loss += loss.detach()
        if step % 500 == 0 or step == args.steps:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            validation = evaluate(model, data["validation"], 64)
            record = {"step": step, "train_loss": training_loss.item() / (500 if step % 500 == 0 else step % 500),
                      "validation_loss": validation, "characters_per_second": step * 1024 / elapsed,
                      "elapsed_seconds": elapsed}
            training_loss.zero_()
            history.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            write(args.output / "last.ckpt", model, moments, step)
            torch.save({"sampling": generator.get_state(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()}, args.output / "last-rng.pt")
            if validation < best:
                best = validation
                write(args.output / "best.ckpt", model, moments, step)
            if step == 500:
                estimate = elapsed * (args.steps - step) / step
                if record["characters_per_second"] < 20000 or estimate > (args.deadline - time.time()) * 0.8:
                    raise RuntimeError("Calibration missed the speed or time limit; checkpoint saved")
    selected, _ = load(args.output / "best.ckpt")
    selected = selected.cuda()
    test = torch.tensor(np.frombuffer((args.data / "test.txt").read_bytes(), dtype=np.uint8).copy(), device="cuda")
    result = {"parameters": 4852992, "backend": "PyTorch 2.8.0 CUDA FP32", "seed": 7,
              "device": torch.cuda.get_device_name(), "updates": args.steps,
              "character_presentations": args.steps * 1024, "selected_update": selected.header[10],
              "validation_loss": evaluate(selected, data["validation"], 1024),
              "test_loss": evaluate(selected, test, 1024), "evaluation_sequences_per_split": 1024,
              "sampled_characters_per_split": 524288, "elapsed_seconds": time.monotonic() - started}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
