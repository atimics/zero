"""Portable row-scaled int8 weights with a checked Python reference loader."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch
from crownless_v2 import Config, Crownless, load

MAGIC = b'CCOREV2\0'


def export(model, tokenizer_path, path, metadata=None):
    payload, tensors = bytearray(), []
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().numpy().astype('<f4')
        item = {'name': name, 'shape': list(array.shape), 'offset': len(payload)}
        if array.ndim == 2:
            scale = np.maximum(np.abs(array).max(axis=1) / 127., 1e-12).astype('<f4')
            quantized = np.clip(np.rint(array / scale[:, None]), -127, 127).astype('i1')
            payload.extend(quantized.tobytes())
            item.update(dtype='i8', scales=len(payload))
            payload.extend(scale.tobytes())
        else:
            item['dtype'] = 'f32'
            payload.extend(array.tobytes())
        tensors.append(item)
    header = {'version': 1, 'config': vars(model.config), 'mode': model.mode, 'tensors': tensors,
              'tokenizer_sha256': hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
              'payload_sha256': hashlib.sha256(payload).hexdigest()}
    header.update({k: v for k, v in (metadata or {}).items() if k in ('meaning_ids', 'kind_ids', 'rules_sha256')})
    encoded = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    Path(path).write_bytes(MAGIC + struct.pack('<I', len(encoded)) + encoded + payload)
    return header


def load_export(path, tokenizer_path, device='cpu'):
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC or len(raw) < 12: raise ValueError('Invalid model export')
    length, = struct.unpack_from('<I', raw, 8)
    if length > 1000000 or length > len(raw) - 12: raise ValueError('Invalid model header')
    header = json.loads(raw[12:12 + length])
    payload = raw[12 + length:]
    if header['version'] != 1: raise ValueError('Unknown export version')
    if hashlib.sha256(payload).hexdigest() != header['payload_sha256']: raise ValueError('Model payload differs')
    if hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest() != header['tokenizer_sha256']:
        raise ValueError('Tokenizer differs from the export')
    config = Config(**header['config'])
    if not (1 <= config.dim <= 256 and 1 <= config.layers <= 16 and 1 <= config.vocab <= 4096 and
            1 <= config.ff <= 2048 and 1 <= config.context <= 512 and 0 <= config.kinds <= 256 and
            1 <= config.heads <= config.dim and config.dim % config.heads == 0 and
            (config.dim // config.heads) % 2 == 0):
        raise ValueError('Export dimensions exceed this runtime')
    model = Crownless(config, header['mode'])
    expected, weights, cursor = model.state_dict(), {}, 0
    for item in header['tensors']:
        name, shape = item['name'], item['shape']
        if name not in expected or name in weights or list(expected[name].shape) != shape:
            raise ValueError('Tensor schema differs')
        if item['offset'] != cursor: raise ValueError('Tensor layout differs')
        count = expected[name].numel()
        if item['dtype'] == 'i8' and len(shape) == 2:
            array = np.frombuffer(payload, dtype='i1', count=count, offset=cursor).reshape(shape).astype('float32')
            cursor += count
            if item['scales'] != cursor: raise ValueError('Scale layout differs')
            scales = np.frombuffer(payload, dtype='<f4', count=shape[0], offset=cursor)
            if not np.all(np.isfinite(scales) & (scales > 0)): raise ValueError('Invalid scale')
            array *= scales[:, None]
            cursor += shape[0] * 4
        elif item['dtype'] == 'f32' and len(shape) == 1:
            array = np.frombuffer(payload, dtype='<f4', count=count, offset=cursor).copy()
            cursor += count * 4
        else:
            raise ValueError('Invalid tensor encoding')
        if not np.isfinite(array).all(): raise ValueError('Non-finite tensor')
        weights[name] = torch.from_numpy(array)
    if cursor != len(payload) or weights.keys() != expected.keys(): raise ValueError('Incomplete model payload')
    model.load_state_dict(weights)
    return model.to(device), header


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    model, metadata = load(args.checkpoint, args.tokenizer)
    export(model, args.tokenizer, args.output, metadata)
    loaded, _ = load_export(args.output, args.tokenizer)
    print(json.dumps({'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'parameters': sum(p.numel() for p in loaded.parameters())}))
