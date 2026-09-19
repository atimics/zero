"""A common target roster and streaming record packs for the 5M experiment."""
import hashlib
import json
import random
import numpy as np


def roster(records, budget, shuffle=False, seed=7):
    if budget <= 0 or not records or any(r['tokens'] < 2 for r in records):
        raise ValueError('Use positive budget and records with at least two tokens')
    result = []; remaining = budget; epoch = 0
    while remaining:
        visits = []
        for i, row in enumerate(records):
            count = min(remaining, row['tokens'] - 1)
            visits.append((i, count)); remaining -= count
            if not remaining:
                break
        if shuffle:
            random.Random(f'boundary-v1:{seed}:{epoch}').shuffle(visits)
        result.extend(visits); epoch += 1
    return result


def roster_digest(visits, canonical=False):
    rows = sorted(visits) if canonical else visits
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def load_records(path, stream_length):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    offset = 0; seen = set()
    for row in records:
        if row['start'] != offset or row['tokens'] < 2 or row['id'] in seen:
            raise ValueError('Record index must be contiguous and unique')
        seen.add(row['id']); offset += row['tokens']
    if offset != stream_length:
        raise ValueError('Record index and stream size differ')
    return records


def packs(data, records, visits, context):
    """Every scored y is within a record; final input in each visit is masked."""
    x = np.zeros(context, dtype=np.int64)
    y = np.full(context, -100, dtype=np.int64)
    segments = np.full(context, -1, dtype=np.int64)
    used = 0
    for visit, (index, count) in enumerate(visits):
        row = records[index]
        if not 1 <= count < row['tokens']:
            raise ValueError('Target visit outside record')
        consumed = 0
        while consumed < count + 1:
            take = min(context - used, count + 1 - consumed)
            start = row['start'] + consumed
            x[used:used + take] = data[start:start + take]
            scored = min(take, max(0, count - consumed))
            y[used:used + scored] = data[start + 1:start + scored + 1]
            segments[used:used + take] = visit
            used += take; consumed += take
            if used == context:
                yield x, y, segments
                x = np.zeros(context, dtype=np.int64)
                y = np.full(context, -100, dtype=np.int64)
                segments = np.full(context, -1, dtype=np.int64); used = 0
    if used:
        yield x, y, segments


def positions_and_mask(segments):
    import torch
    b, length = segments.shape
    indices = torch.arange(length, device=segments.device).expand(b, -1)
    starts = torch.ones_like(segments, dtype=torch.bool)
    starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
    anchors = torch.where(starts, indices, 0).cummax(dim=1).values
    positions = indices - anchors
    causal = indices[0][None, :] <= indices[0][:, None]
    mask = (segments[:, :, None] == segments[:, None, :]) & causal
    return positions, mask[:, None]


def forward(model, tokens, segments, isolate, dropout=0.):
    if not isolate:
        return model(tokens, dropout=dropout)
    import torch
    from torch.nn import functional as F
    positions, mask = positions_and_mask(segments)
    def rotate(value):
        value = value.reshape(*tokens.shape, model.heads, -1).transpose(1, 2)
        even, odd = value[..., 0::2], value[..., 1::2]
        c = model.rope_cos[positions].to(even.dtype)[:, None]
        s = model.rope_sin[positions].to(even.dtype)[:, None]
        return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)
    value = F.embedding(tokens, model.weights[0])
    for offset in range(1, len(model.weights)-1, 8):
        n1, wq, wk, wv, wo, n2, w1, w2 = model.weights[offset:offset+8]
        normalized = model.norm(value, n1)
        q, k = rotate(F.linear(normalized, wq)), rotate(F.linear(normalized, wk))
        v = F.linear(normalized, wv).reshape(*tokens.shape, model.heads, -1).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attention = attention.transpose(1, 2).reshape(*tokens.shape, model.dim)
        value = value + F.dropout(F.linear(attention, wo), dropout, model.training)
        hidden = F.gelu(F.linear(model.norm(value, n2), w1), approximate='tanh')
        value = value + F.dropout(F.linear(hidden, w2), dropout, model.training)
    return F.linear(model.norm(value, model.weights[-1]), model.weights[0])
