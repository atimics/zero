"""The Shannon loop: the model proposes, the world verifies, and only verified gains are trained in.

  python shannon_loop.py --start runs/<run>/output/best-mining.ckpt --arm checked --output loops/checked

Each round:
  1. Fresh flight states, never used before (seeds by round).
  2. Propose: the model's greedy control plus PROPOSALS-1 more sampled from its own distribution.
  3. Verify: signal_replay flies only those proposals; Signal's utility scores them.
  4. Keep: each state's best verified proposal becomes a training row.
  5. Retrain: FT_STEPS of fine-tuning on the kept rows, half of every batch replayed from the six original
     streams so the voices and the dragon are not forgotten.
  6. Gate: the checked arm keeps the new model only if mean utility on the gate states does not fall.
     The unchecked arm keeps every new model.
The audit states are scored every round and never decide anything. All 9 utilities of each gate and audit
state come from data/mining.valid.utility.jsonl, so the judge there is exact and costs no simulation.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
sys.path.insert(0, str(HERE))
from zero_torch import load, update, write  # noqa: E402
import build_data  # noqa: E402
from train_cuda import STREAMS, VOICES, as_tensor, judge_dragon, stream_loss  # noqa: E402

ACTIONS = build_data.ACTIONS


@torch.no_grad()
def action_logprobs(model, prompts, device):
    """log p(action | prompt) over the nine controls for each prompt, normalized over the nine."""
    texts = [f"\n{p} > {a}\n".encode() for p in prompts for a in ACTIONS]
    heads = [len(f"\n{p} >".encode()) for p in prompts for _ in ACTIONS]
    width = max(map(len, texts))
    tokens = torch.zeros((len(texts), width), dtype=torch.long)
    for i, t in enumerate(texts):
        tokens[i, :len(t)] = torch.tensor(list(t))
    tokens = tokens.to(device)
    scores = torch.empty(len(texts))
    for lo in range(0, len(texts), 288):
        chunk = tokens[lo:lo + 288]
        logp = F.log_softmax(model(chunk[:, :-1]).float(), -1)
        gathered = logp.gather(-1, chunk[:, 1:, None])[..., 0].cpu()
        for j in range(len(chunk)):
            i = lo + j
            scores[i] = gathered[j, heads[i] - 1:len(texts[i]) - 1].sum()
    return torch.log_softmax(scores.view(len(prompts), len(ACTIONS)), -1)


def judge(model, rows, device):
    """Mean Signal utility of the model's greedy control, and how often it is the best control."""
    logp = action_logprobs(model, [r["prompt"] for r in rows], device)
    picks = logp.argmax(-1).tolist()
    utility = [r["utility"][ACTIONS[p]] for r, p in zip(rows, picks)]
    best = [r["utility"][ACTIONS[p]] >= max(r["utility"].values()) for r, p in zip(rows, picks)]
    return {"utility": float(np.mean(utility)), "picked_best": float(np.mean(best)),
            "catastrophes": int(sum(u < -5 for u in utility))}


def propose_and_verify(model, binary, seeds, proposals, rng, device, workers):
    states = [build_data.flight_state(binary, s) for s in seeds]
    logp = action_logprobs(model, [p for p, _ in states], device)
    jobs = []
    for (prompt, cmd), row in zip(states, logp):
        greedy = int(row.argmax())
        picks = [greedy]
        weights = row.exp().numpy().copy()
        weights[greedy] = 0
        while len(picks) < proposals and weights.sum() > 0:
            k = rng.choices(range(len(ACTIONS)), weights=weights)[0]
            picks.append(k)
            weights[k] = 0
        jobs.append((prompt, cmd, [ACTIONS[k] for k in picks]))
    with ThreadPoolExecutor(workers) as pool:
        verified = list(pool.map(lambda j: build_data.fly(j[1], j[2]), jobs))
    kept, improved, gains = [], 0, []
    for (prompt, _, names), utility in zip(jobs, verified):
        best = max(utility, key=utility.get)
        kept.append(f"{prompt} > {best}")
        improved += best != names[0]
        gains.append(utility[best] - utility[names[0]])
    return kept, improved / len(jobs), float(np.mean(gains))


def finetune(model, moments, step, kept, replay, steps, batch, window, lr, gen, device):
    text = torch.tensor(list(("\n".join(kept) + "\n").encode()), dtype=torch.uint8, device=device)
    offsets = torch.arange(window + 1, device=device)
    model.train()
    for _ in range(steps):
        step += 1
        rows = []
        half = batch // 2
        s = torch.randint(len(text) - window - 1, (half,), generator=gen, device=device)
        rows.append(text[s[:, None] + offsets])
        for _ in range(batch - half):
            data = replay[int(torch.randint(len(replay), (1,), generator=gen, device=device))]
            s = torch.randint(len(data) - window - 1, (1,), generator=gen, device=device)
            rows.append(data[s[:, None] + offsets])
        w = torch.cat(rows).long()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(w[:, :-1], dropout=0.1).flatten(0, 1), w[:, 1:].flatten())
        loss.backward()
        update(model, moments, step, lr)
    model.eval()
    return step


def snapshot(model, moments):
    return ([w.detach().clone() for w in model.weights], [[m.clone(), v.clone()] for m, v in moments])


def restore(model, moments, saved):
    weights, saved_moments = saved
    with torch.no_grad():
        for w, s in zip(model.weights, weights):
            w.copy_(s)
    for (m, v), (sm, sv) in zip(moments, saved_moments):
        m.copy_(sm)
        v.copy_(sv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--arm", choices=["checked", "unchecked"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--signal-replay", default=str(HERE.parents[2] / "signal/build/signal_replay"))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--states", type=int, default=1500)
    ap.add_argument("--proposals", type=int, default=3)
    ap.add_argument("--ft-steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    model, raw = load(args.start)
    model = model.to(device).eval()
    moments = [[torch.from_numpy(a).reshape(w.shape).to(device) for a in pair] for w, pair in zip(model.weights, raw)]
    step = model.header[10]
    judged = [json.loads(l) for l in (args.data / "mining.valid.utility.jsonl").read_text().splitlines() if l]
    gate, audit = judged[: len(judged) // 2], judged[len(judged) // 2:]
    replay = [as_tensor(args.data / f"{s}.train.txt", device) for s in STREAMS]
    valid = {s: as_tensor(args.data / f"{s}.valid.txt", device) for s in VOICES + ["dragon"]}

    def report(round_index, accepted, extra):
        record = {"round": round_index, "arm": args.arm, "accepted": accepted, "gate": judge(model, gate, device),
                  "audit": judge(model, audit, device),
                  "voice_loss": {s: stream_loss(model, valid[s], 16) for s in VOICES},
                  "dragon": judge_dragon(model, args.data / "dragon.valid.txt", device, 400)["matches_rule"],
                  **extra, "seconds": time.monotonic() - started}
        print(json.dumps(record), flush=True)
        with (args.output / "rounds.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    started = time.monotonic()
    (args.output / "rounds.jsonl").write_text("")
    current = report(0, True, {})
    for r in range(1, args.rounds + 1):
        seeds = range(20_000_000 + r * 100_000, 20_000_000 + r * 100_000 + args.states)
        kept, improved, gain = propose_and_verify(model, args.signal_replay, seeds, args.proposals, rng, device,
                                                  args.workers)
        saved = snapshot(model, moments)
        step = finetune(model, moments, step, kept, replay, args.ft_steps, args.batch, args.window, args.lr, gen,
                        device)
        candidate_gate = judge(model, gate, device)["utility"]
        accept = args.arm == "unchecked" or candidate_gate >= current["gate"]["utility"]
        if not accept:
            restore(model, moments, saved)
        current = report(r, accept, {"proposal_beat_greedy": improved, "mean_verified_gain": gain,
                                     "candidate_gate_utility": candidate_gate})
    write(args.output / "final.ckpt", model, moments, step)


if __name__ == "__main__":
    main()
