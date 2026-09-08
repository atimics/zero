"""Score one checkpoint on the frozen 1,024-window subword protocol."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from score_subword_windows import score
from subword_model import create


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = create(state['config'])
    model.load_state_dict(state['model'])
    model.eval()
    lengths = torch.tensor(json.loads((a.data / 'token_bytes.json').read_text()))
    result = {'backend': 'cpu fp32', 'step': state['step'], 'config': state['config'],
              'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(), 'splits': {}, 'scores': {}}
    for split in ['validation', 'test']:
        data = torch.tensor(np.fromfile(a.data / (split + '.bin'), dtype='<u2').astype('int32'))
        rows = score(model, data, lengths, 1024, contextlib.nullcontext)
        result['splits'][split] = rows
        total_bytes = sum(r['bytes'] for r in rows)
        result['scores'][split] = {'bits_per_byte': sum(r['nats'] for r in rows) / math.log(2) / total_bytes, 'bytes': total_bytes, 'windows': len(rows)}
        a.output.write_text(json.dumps(result, indent=2) + '\n')
        print(split, result['scores'][split], flush=True)

if __name__ == '__main__':
    main()
