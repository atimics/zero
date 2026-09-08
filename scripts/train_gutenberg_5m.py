#!/usr/bin/env python3
"""Run the fixed local training plan, then export and evaluate its best checkpoint."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan["repository"])
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "status.json"
    if state_path.exists():
        raise SystemExit("This run already has a status file. Use a new run directory.")
    state = {"phase": "checking", "pid": os.getpid(), "started_at": time.time(),
             "plan_sha256": sha(args.plan), "parameters": 4852992}
    save(state_path, state)
    try:
        for item in plan["inputs"] + plan["binaries"]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"Input hash differs: {item['path']}")
        environment = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1"}
        state["phase"] = "training"
        save(state_path, state)
        command = [str(root / "literary_lm"), *plan["training_arguments"]]
        with (output / "training.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           cwd=root, env=environment, check=True)
        log = (output / "training.log").read_text()
        updates = re.findall(r"^update\s+(\d+) train ([\d.]+) val ([\d.]+)", log, re.M)
        if not updates or int(updates[-1][0]) != plan["steps"]:
            raise ValueError("Training ended before the planned update count")
        state["phase"] = "evaluating"
        state["last_update"] = int(updates[-1][0])
        save(state_path, state)
        best = output / "best.ckpt"
        exported = output / "gutenberg-5m.litq8"
        subprocess.run([str(root / "export_literary"), str(best), str(exported)], check=True)
        evaluations = {}
        for split in ["validation", "test"]:
            result = subprocess.run([str(root / "literary_lm"), "--resume", str(best),
                "--steps", "0", "--evaluate", plan["texts"][split], "--validation", "1024",
                "--tokens", "0"], text=True, capture_output=True, env=environment, check=True)
            (output / f"{split}-evaluation.log").write_text(result.stdout)
            match = re.search(r"evaluation tokens=(\d+) sequences=(\d+) loss=([\d.]+)", result.stdout)
            if not match:
                raise ValueError("Evaluation result missing")
            evaluations[split] = {"corpus_characters": int(match[1]), "sequences": int(match[2]),
                                  "sampled_characters": int(match[2]) * 512, "loss": float(match[3])}
        samples = []
        for name, model in [("gutenberg-5m", exported), ("warrenmind", root / "docs/model.litq8")]:
            for prompt in plan["prompts"]:
                start = time.monotonic()
                result = subprocess.run([str(root / "literary_infer"), str(model), prompt, "320"],
                                        capture_output=True, text=True, check=True)
                samples.append({"model": name, "prompt": prompt, "output": result.stdout,
                                "seconds": time.monotonic() - start})
        save(output / "samples.json", samples)
        result = {"parameters": 4852992, "context": 512, "updates": state["last_update"],
                  "character_presentations": state["last_update"] * 512 * plan["batch"],
                  "evaluations": evaluations, "model_sha256": sha(exported),
                  "best_checkpoint_sha256": sha(best), "last_checkpoint_sha256": sha(output / "last.ckpt"),
                  "training_seconds": time.time() - state["started_at"]}
        save(output / "result.json", result)
        state.update({"phase": "complete", "finished_at": time.time(), "result": "result.json"})
        save(state_path, state)
    except BaseException as error:
        state.update({"phase": "failed", "error": str(error), "finished_at": time.time()})
        save(state_path, state)
        raise


if __name__ == "__main__":
    main()
