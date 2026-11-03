#!/usr/bin/env python3
"""Build the six training streams for the generalist model.

Four voices (Blake, Shakespeare, Crowley, Crownless) and two worlds
(Signal's asteroid field and the crownless2 dragon realm). Every record
starts with its tag, so any training window says which stream it is from.

  python3 build_data.py --signal-replay PATH --cc2trace PATH --crownless PATH

Writes data/<stream>.train.txt and data/<stream>.valid.txt (ASCII only).
World validation rows come from seeds the training rows never use.
"""

import argparse
import json
import math
import random
import re
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZERO = HERE.parent.parent
ACTIONS = ["NONE", "W", "A", "D", "S", "WA", "WD", "SA", "SD"]
CHUNK = 420


def ascii_text(text):
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", "--").replace("–", "-").replace("…", "...")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\x0a\x20-\x7e]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def chunks(text, tag):
    """Split on blank lines, then pack paragraphs into tagged records of about CHUNK characters."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        para = para.strip("\n")
        if not para.strip():
            continue
        if cur and len(cur) + len(para) > CHUNK:
            out.append(f"<{tag}>\n{cur}\n")
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
        while len(cur) > 2 * CHUNK:
            cut = cur.rfind("\n", 0, CHUNK) + 1 or CHUNK
            out.append(f"<{tag}>\n{cur[:cut].rstrip()}\n")
            cur = cur[cut:]
    if cur:
        out.append(f"<{tag}>\n{cur}\n")
    return out


def split_write(name, records, valid_every=20):
    train = [r for i, r in enumerate(records) if i % valid_every]
    valid = [r for i, r in enumerate(records) if i % valid_every == 0]
    write(name, train, valid)


def write(name, train, valid):
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / f"{name}.train.txt").write_text("\n".join(train) + "\n")
    (data / f"{name}.valid.txt").write_text("\n".join(valid) + "\n")
    print(f"{name}: {len(train)} train, {len(valid)} valid, {sum(map(len, train)) // 1024} KiB")


def crownless_lines(root):
    """Authored Crownless speech: the held-account corpus plus spoken lines in the story sources."""
    lines = []
    corpus = root / "out/core-v2-data/train.txt"
    if corpus.exists():
        lines += [b for b in corpus.read_text().split("\n\n") if b.strip()]
    for name in ["cc_gate_voice.c", "cc_road_voice.c", "cc_speech_interactions.c", "cc_story.c", "cc_hrakhor.c"]:
        src = (root / "src/story" / name).read_text()
        for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"(?:\s*"((?:[^"\\\n]|\\.)*)")*', src):
            s = re.sub(r'"\s*"', "", m.group(0))[1:-1].replace('\\"', '"').replace("\\n", " ").strip()
            if len(s) >= 25 and " " in s and not re.search(r"[{}<>=_/]|%[^sd]", s):
                lines.append(s.replace("%s", "Thornford").replace("%d", "3"))
    for doc in ["hrakhor-corpus.md", "fallen-purses.md", "crown-carriage-roads.md"]:
        path = root / "docs" / doc
        if path.exists():
            text = re.sub(r"```[a-z]*\n?", "", path.read_text())
            lines += [p for p in text.split("\n\n") if p.strip() and not p.startswith(("#", "|", "-  ", "<"))]
    random.Random(7).shuffle(lines)
    return chunks(ascii_text("\n\n".join(lines)), "crownless")


def mining_row(binary, seed):
    """One random flight state; try all nine controls for one second; label the best by Signal's own utility."""
    rng = random.Random(seed)
    spawn = (rng.uniform(-2500, 2500), rng.uniform(-2500, 2500))
    angle = rng.uniform(-math.pi, math.pi)
    vel = (rng.uniform(-150, 150), rng.uniform(-150, 150))
    dist, bearing = rng.uniform(200, 2500), rng.uniform(-math.pi, math.pi)
    goal = (spawn[0] + dist * math.cos(bearing), spawn[1] + dist * math.sin(bearing))
    cmd = [binary, "--seed", str(2037 + seed % 7), "--spawn", f"{spawn[0]:.1f},{spawn[1]:.1f}",
           "--velocity", f"{vel[0]:.1f},{vel[1]:.1f}", "--angle", f"{angle:.4f}",
           "--goal", f"{goal[0]:.1f},{goal[1]:.1f}", "--history", "NONE", "--horizon-ticks", "120",
           "--candidates", ",".join(ACTIONS), "--out", "/dev/stdout"]
    rows = [json.loads(l) for l in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines()
            if l.startswith("{")]
    best = max(rows, key=lambda r: r["utility"])
    # Ship frame: the goal's bearing off the nose, and velocity along and across the nose.
    rel = math.degrees(math.atan2(goal[1] - spawn[1], goal[0] - spawn[0]) - angle)
    rel = (rel + 180) % 360 - 180
    along = vel[0] * math.cos(angle) + vel[1] * math.sin(angle)
    across = -vel[0] * math.sin(angle) + vel[1] * math.cos(angle)
    return (f"<mining> d={dist:.0f} b={rel:+.0f} v={along:+.0f},{across:+.0f} > {best['candidate_name']}",
            {r["candidate_name"]: r["utility"] for r in rows})


def mining(binary, train_n, valid_n, workers):
    with ThreadPoolExecutor(workers) as pool:
        train = list(pool.map(lambda s: mining_row(binary, s), range(train_n)))
        valid = list(pool.map(lambda s: mining_row(binary, 10_000_000 + s), range(valid_n)))
    write("mining", [t for t, _ in train], [t for t, _ in valid])
    # The judge for later: every candidate's utility for each validation row.
    (HERE / "data/mining.valid.utility.jsonl").write_text(
        "".join(json.dumps({"prompt": t.rsplit(" > ", 1)[0], "utility": u}) + "\n" for t, u in valid))


def dragon(binary, worlds, keep):
    def run(seed0, n):
        out = subprocess.run([binary, "rabbit", str(n), "1", str(seed0), str(keep)],
                             capture_output=True, text=True, check=True).stdout
        return [l for l in out.splitlines() if l]
    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda s: run(s, 1), range(1, worlds + 1)))
    train = [l for p in parts for l in p]
    valid = run(100_001, 2)
    random.Random(3).shuffle(train)
    write("dragon", train, valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-replay", required=True)
    ap.add_argument("--cc2trace", required=True)
    ap.add_argument("--crownless", required=True, type=Path)
    ap.add_argument("--mining-train", type=int, default=6000)
    ap.add_argument("--mining-valid", type=int, default=500)
    ap.add_argument("--dragon-worlds", type=int, default=16)
    ap.add_argument("--dragon-keep", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    for name in ["blake", "shakespeare", "crowley"]:
        split_write(name, chunks(ascii_text((ZERO / "corpus" / f"{name}.txt").read_text()), name))
    split_write("crownless", crownless_lines(args.crownless))
    dragon(args.cc2trace, args.dragon_worlds, args.dragon_keep)
    mining(args.signal_replay, args.mining_train, args.mining_valid, args.workers)


if __name__ == "__main__":
    main()
