"""Frozen source and atomic evidence helpers for boundary experiments."""
import hashlib
import json
import os
from pathlib import Path
from canada_narrative import ROOT, digest, read_json

EXPERIMENT = ROOT / 'experiments/small-model-boundaries-v1'


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def contract():
    from run_canada_narrative import verify_code
    verify_code()
    lock = read_json(EXPERIMENT / 'source.lock.json')
    for name, expected in lock.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Boundary source identity mismatch: ' + name)
    return read_json(EXPERIMENT / 'contract.json')
