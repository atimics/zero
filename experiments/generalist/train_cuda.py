"""Train one ZERO literary model on all six streams, then score it on the voices and both worlds.

  python train_cuda.py --data data --initial initial.ckpt --output output --steps 30000

Each batch row picks a stream uniformly, then a random 513-byte window in it, so Blake's 59 KiB counts
as much as Shakespeare's 5 MiB. Checkpoints are native C files (scripts/zero_torch.py), so literary_lm,
export_literary and literary_infer read them unchanged.

At the end it writes result.json:
  per-stream held-out loss;
  mining: on held-out Signal states, the control the model ranks first, scored by Signal's own utility
          for that control, against the best control, the average control, and always-thrust;
  dragon: on held-out crownless2 worlds, how often the model's goal matches the rule's, against always-majority;
and samples.json: 600 characters from each voice tag.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from zero_torch import load, update, write  # noqa: E402

STREAMS = ["blake", "shakespeare", "crowley", "crownless", "mining", "dragon"]
VOICES = ["blake", "shakespeare", "crowley", "crownless"]
GOAL_LETTERS = {"G": "graze", "S": "shelter", "H": "hunt", "Z": "sleep", "B": "breed"}
WINDOW = 512


def as_tensor(path, device):
    return torch.tensor(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy(), device=device)


@torch.no_grad()
def stream_loss(model, data, windows):
    model.eval()
    starts = torch.arange(windows, device=data.device) * (len(data) - WINDOW - 1) // max(1, windows - 1)
    total = 0.0
    for group in starts.split(16):
        w = data[group[:, None] + torch.arange(WINDOW + 1, device=data.device)].long()
        total += F.cross_entropy(model(w[:, :-1]).flatten(0, 1), w[:, 1:].flatten()).item() * len(group)
    model.train()
    return total / windows


@torch.no_grad()
def rank(model, prompt, candidates, device):
    """Sum of log-probabilities of ' > CANDIDATE\\n' after the prompt, for each candidate, in one batch."""
    texts = [f"\n{prompt} > {c}\n".encode() for c in candidates]
    head = len(f"\n{prompt} >".encode())
    width = max(map(len, texts))
    tokens = torch.zeros((len(texts), width), dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        tokens[i, :len(t)] = torch.tensor(list(t), device=device)
    logp = F.log_softmax(model(tokens[:, :-1]).float(), -1)
    scores = []
    for i, t in enumerate(texts):
        positions = torch.arange(head, len(t), device=device)
        scores.append(logp[i, positions - 1, tokens[i, positions]].sum().item())
    return scores


@torch.no_grad()
def judge_mining(model, path, device):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l]
    chosen, best, mean, thrust, hits = [], [], [], [], 0
    for r in rows:
        names = list(r["utility"])
        pick = names[int(np.argmax(rank(model, r["prompt"], names, device)))]
        u = r["utility"]
        top = max(u, key=u.get)
        chosen.append(u[pick]); best.append(u[top]); mean.append(sum(u.values()) / len(u)); thrust.append(u["W"])
        hits += pick == top
    return {"rows": len(rows), "picked_best": hits / len(rows), "model_utility": float(np.mean(chosen)),
            "best_utility": float(np.mean(best)), "average_control_utility": float(np.mean(mean)),
            "always_thrust_utility": float(np.mean(thrust))}


@torch.no_grad()
def judge_dragon(model, path, device, limit):
    lines = [l for l in path.read_text().splitlines() if " > " in l]
    lines = lines[:: max(1, len(lines) // limit)][:limit]
    hits, majority, counts = 0, 0, {}
    for l in lines:
        prompt, label = l.rsplit(" > ", 1)
        counts[label] = counts.get(label, 0) + 1
        legal = [GOAL_LETTERS[c] for c in prompt.rsplit("ok=", 1)[1] if c in GOAL_LETTERS]
        if len(legal) == 1:
            pick = legal[0]
        else:
            pick = legal[int(np.argmax(rank(model, prompt, legal, device)))]
        hits += pick == label
    top = max(counts, key=counts.get)
    majority = counts[top] / len(lines)
    return {"rows": len(lines), "matches_rule": hits / len(lines), "always_majority": majority,
            "majority_goal": top, "label_counts": counts}


@torch.no_grad()
def sample(model, tag, length, device, generator):
    model.eval()
    tokens = torch.tensor([list(f"<{tag}>\n".encode())], device=device)
    out = []
    for _ in range(length):
        logits = model(tokens[:, -WINDOW:])[0, -1].float() / 0.8
        top = torch.topk(logits, 40)
        nxt = top.indices[torch.multinomial(F.softmax(top.values, -1), 1, generator=generator)]
        out.append(int(nxt))
        tokens = torch.cat([tokens, nxt.view(1, 1)], 1)
    model.train()
    text = bytes(out).decode("ascii", "replace")
    return text.split("\n<", 1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--initial", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--deadline", type=float, default=float("inf"), help="unix time; stop and score before it")
    ap.add_argument("--dragon-rows", type=int, default=3000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    model, raw = load(args.initial)
    model = model.to(device)
    moments = [[torch.from_numpy(a).reshape(w.shape).to(device) for a in pair] for w, pair in zip(model.weights, raw)]
    train = {s: as_tensor(args.data / f"{s}.train.txt", device) for s in STREAMS}
    valid = {s: as_tensor(args.data / f"{s}.valid.txt", device) for s in STREAMS}
    for s, t in {**train, **{s + " valid": v for s, v in valid.items()}}.items():
        if int(t.max()) >= model.vocab:
            raise ValueError(f"{s} has a byte outside the vocabulary")
    gen = torch.Generator(device=device).manual_seed(7)
    lengths = torch.tensor([len(train[s]) - WINDOW - 1 for s in STREAMS], device=device)
    joined = torch.cat([train[s] for s in STREAMS])
    bases = torch.tensor([0] + [len(train[s]) for s in STREAMS[:-1]], device=device).cumsum(0)
    offsets = torch.arange(WINDOW + 1, device=device)
    history, best, started, running = [], float("inf"), time.monotonic(), torch.zeros((), device=device)
    step = 0
    for step in range(1, args.steps + 1):
        if time.time() > args.deadline - 900:
            print("stopping early to leave time for scoring", flush=True)
            break
        which = torch.randint(len(STREAMS), (args.batch,), generator=gen, device=device)
        starts = bases[which] + (torch.rand(args.batch, generator=gen, device=device) * lengths[which]).long()
        w = joined[starts[:, None] + offsets].long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(w[:, :-1], dropout=0.1).flatten(0, 1), w[:, 1:].flatten())
        loss.backward()
        lr = args.lr * min(1.0, step / args.warmup)
        if step > args.warmup:
            lr *= 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        update(model, moments, step, lr)
        running += loss.detach()
        if step % args.report == 0 or step == args.steps:
            losses = {s: stream_loss(model, valid[s], 32) for s in STREAMS}
            mean = sum(losses.values()) / len(losses)
            record = {"step": step, "train_loss": running.item() / args.report, "valid_loss": losses,
                      "valid_mean": mean, "seconds": time.monotonic() - started}
            running.zero_()
            history.append(record)
            print(json.dumps(record), flush=True)
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            write(args.output / "last.ckpt", model, moments, step)
            if mean < best:
                best = mean
                write(args.output / "best.ckpt", model, moments, step)
    model, _ = load(args.output / "best.ckpt")
    model = model.to(device).eval()
    result = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
              "parameters": sum(w.numel() for w in model.weights), "updates": step, "batch": args.batch,
              "selected_update": model.header[10],
              "valid_loss": {s: stream_loss(model, valid[s], 128) for s in STREAMS},
              "mining": judge_mining(model, args.data / "mining.valid.utility.jsonl", device),
              "dragon": judge_dragon(model, args.data / "dragon.valid.txt", device, args.dragon_rows),
              "seconds": time.monotonic() - started}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    g = torch.Generator(device=device).manual_seed(11)
    samples = {v: [sample(model, v, 600, device, g) for _ in range(2)] for v in VOICES}
    (args.output / "samples.json").write_text(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
