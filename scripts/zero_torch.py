"""PyTorch execution of ZERO's rotary literary architecture and C checkpoint format."""
import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HEADER = struct.Struct("<8s9I4x2Q")


class Zero(nn.Module):
    def __init__(self, header, arrays):
        super().__init__()
        self.header = list(header)
        _, version, vocab, context, dim, heads, layers, ff, count, rotary, step, rng = header
        if version != 3 or rotary != 1 or count != 2 + layers * 8:
            raise ValueError("Expected a rotary ZERO v3 checkpoint")
        self.vocab, self.context, self.dim, self.heads = vocab, context, dim, heads
        shapes = [(vocab, dim)]
        for _ in range(layers):
            shapes.extend([(dim,), (dim, dim), (dim, dim), (dim, dim), (dim, dim), (dim,), (ff, dim), (dim, ff)])
        shapes.append((dim,))
        self.weights = nn.ParameterList([nn.Parameter(torch.from_numpy(a.copy()).reshape(shape))
                                        for a, shape in zip(arrays, shapes, strict=True)])
        frequency = 10000.0 ** (-torch.arange(0, dim // heads, 2).float() / (dim // heads))
        angles = torch.arange(context).float()[:, None] * frequency[None, :]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def rotate(self, value):
        b, t, _ = value.shape
        value = value.reshape(b, t, self.heads, -1).transpose(1, 2)
        even, odd = value[..., 0::2], value[..., 1::2]
        c, s = self.rope_cos[:t], self.rope_sin[:t]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)

    @staticmethod
    def norm(value, weight):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * weight

    def forward(self, tokens, dropout=0.0):
        value = F.embedding(tokens, self.weights[0])
        for offset in range(1, len(self.weights) - 1, 8):
            n1, wq, wk, wv, wo, n2, w1, w2 = self.weights[offset:offset + 8]
            normalized = self.norm(value, n1)
            q, k = self.rotate(F.linear(normalized, wq)), self.rotate(F.linear(normalized, wk))
            v = F.linear(normalized, wv).reshape(*tokens.shape, self.heads, -1).transpose(1, 2)
            attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attention = attention.transpose(1, 2).reshape(*tokens.shape, self.dim)
            value = value + F.dropout(F.linear(attention, wo), dropout, self.training)
            hidden = F.gelu(F.linear(self.norm(value, n2), w1), approximate="tanh")
            value = value + F.dropout(F.linear(hidden, w2), dropout, self.training)
        return F.linear(self.norm(value, self.weights[-1]), self.weights[0])


def load(path):
    with open(path, "rb") as stream:
        header = HEADER.unpack(stream.read(HEADER.size))
        if header[0] != b"ZEROLM2\0":
            raise ValueError("Invalid checkpoint magic")
        arrays, moments = [], []
        for _ in range(header[8]):
            count, = struct.unpack("<Q", stream.read(8))
            values = [np.frombuffer(stream.read(count * 4), dtype="<f4").copy() for _ in range(3)]
            if any(len(a) != count for a in values):
                raise ValueError("Truncated checkpoint")
            arrays.append(values[0])
            moments.append(values[1:])
        if stream.read(1):
            raise ValueError("Trailing checkpoint data")
    return Zero(header, arrays), moments


def write(path, model, moments, step):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    header = model.header.copy()
    header[10] = step
    with temporary.open("wb") as stream:
        stream.write(HEADER.pack(*header))
        for weight, (m, v) in zip(model.weights, moments, strict=True):
            stream.write(struct.pack("<Q", weight.numel()))
            for tensor in [weight, m, v]:
                stream.write(tensor.detach().cpu().numpy().astype("<f4").tobytes())
    temporary.replace(path)


@torch.no_grad()
def update(model, moments, step, lr, weight_decay=0.01, clip=1.0):
    # Match the C trainer's bias correction, epsilon, decay groups and clipping.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    correction = lr * math.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
    for index, (weight, (m, v)) in enumerate(zip(model.weights, moments, strict=True)):
        m.mul_(0.9).add_(weight.grad, alpha=0.1)
        v.mul_(0.999).addcmul_(weight.grad, weight.grad, value=0.001)
        decay = index == 0 or (0 < index < len(model.weights) - 1 and (index - 1) % 8 not in [0, 5])
        weight.add_(weight * (-lr * weight_decay if decay else 0))
        weight.addcdiv_(m, v.sqrt().add_(1e-8), value=-correction)
    return norm
