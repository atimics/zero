"""Crownless 5M v2: text, field, and source-copy controls on one backbone."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


@dataclass
class Config:
    vocab: int = 4096
    dim: int = 192
    layers: int = 8
    heads: int = 6
    ff: int = 624
    context: int = 512


def train_tokenizer(rows, path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=['[EOS]'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tokenizer.train_from_iterator((r['prefix'] + r['output'] + '\n' for r in rows), trainer)
    tokenizer.save(str(path))
    return tokenizer


def encode_parts(tokenizer, text, spans):
    raw, tokens, mapped, at = text.encode(), [], [], 0
    for span in sorted(spans, key=lambda s: s['start']):
        start, end = span['start'], span['end']
        if not (at <= start <= end <= len(raw)): raise ValueError('Invalid or overlapping byte spans')
        if raw[start:end].decode() != span['text']: raise ValueError('Source span differs from its text')
        tokens.extend(tokenizer.encode(raw[at:start].decode()).ids)
        first = len(tokens)
        tokens.extend(tokenizer.encode(raw[start:end].decode()).ids)
        mapped.append({**span, 'token_start': first, 'token_end': len(tokens)})
        at = end
    tokens.extend(tokenizer.encode(raw[at:].decode()).ids)
    if tokenizer.decode(tokens) != text: raise ValueError('Tokenizer round trip failed')
    return tokens, mapped


def encode_row(tokenizer, row, context=512):
    prefix, fields = encode_parts(tokenizer, row['prefix'], row['fields'])
    output, copies = encode_parts(tokenizer, row['output'], row['copies'])
    tokens = prefix + output + [tokenizer.token_to_id('[EOS]')]
    if len(tokens) > context + 1: raise ValueError(f"Example exceeds context: {row['id']}")
    n = len(tokens) - 1
    meta = [[0, 0, 0, 0] for _ in range(n)]
    candidates, source_ids, field_to_candidate = [], [], {}
    for field in fields:
        for i in range(field['token_start'], field['token_end']):
            meta[i] = [field['role'], field['knowledge'], field['provenance'], field['event']]
        if field['role'] not in (0, 7, 8) and field['knowledge'] != 3:
            field_to_candidate[field['field']] = len(candidates)
            candidates.append([field['token_start'], field['token_end'] - 1])
            source_ids.append(prefix[field['token_start']:field['token_end']])
    labels = [-100] * (len(prefix) - 1) + tokens[len(prefix):]
    copy_targets, weights = [-1] * n, [1.] * n
    copy_labels = list(labels)
    for copied in copies:
        start = len(prefix) + copied['token_start'] - 1
        end = len(prefix) + copied['token_end'] - 1
        slot = field_to_candidate[copied['field']]
        if output[copied['token_start']:copied['token_end']] != source_ids[slot]:
            raise ValueError('Copy token sequence differs from source')
        copy_targets[start], weights[start] = slot, float(end - start)
        for i in range(start + 1, end): copy_labels[i] = -100
    return {'tokens': tokens[:-1], 'meta': meta, 'labels': labels, 'copy_labels': copy_labels,
            'copy_targets': copy_targets, 'weights': weights, 'candidates': candidates,
            'source_ids': source_ids, 'prefix_length': len(prefix), 'row': row}


def batch(records, device):
    length = max(len(r['tokens']) for r in records)
    slots = max(1, max(len(r['candidates']) for r in records))
    b = len(records)
    out = {'tokens': torch.zeros(b, length, dtype=torch.long),
           'meta': torch.zeros(b, length, 4, dtype=torch.long),
           'labels': torch.full((b, length), -100, dtype=torch.long),
           'copy_labels': torch.full((b, length), -100, dtype=torch.long),
           'copy_targets': torch.full((b, length), -1, dtype=torch.long),
           'weights': torch.ones(b, length),
           'candidates': torch.zeros(b, slots, 2, dtype=torch.long),
           'candidate_mask': torch.zeros(b, slots, dtype=torch.bool)}
    for i, record in enumerate(records):
        n, k = len(record['tokens']), len(record['candidates'])
        for name in ('tokens', 'meta', 'labels', 'copy_labels', 'copy_targets', 'weights'):
            out[name][i, :n] = torch.tensor(record[name])
        if k:
            out['candidates'][i, :k] = torch.tensor(record['candidates'])
            out['candidate_mask'][i, :k] = True
    return {key: value.to(device) for key, value in out.items()}


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c.dim), nn.RMSNorm(c.dim)
        self.qkv = nn.Linear(c.dim, c.dim * 3, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
        self.up = nn.Linear(c.dim, c.ff * 2, bias=False)
        self.down = nn.Linear(c.ff, c.dim, bias=False)
        self.heads = c.heads

    def forward(self, x, cosine, sine, cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [y.view(b, t, self.heads, -1).transpose(1, 2) for y in (q, k, v)]
        def rotate(y):
            a, z = y[..., 0::2], y[..., 1::2]
            return torch.stack((a * cosine - z * sine, a * sine + z * cosine), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        if cache is not None:
            k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        if cache is not None and t > 1:
            past = k.shape[2] - t
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= past + torch.arange(t, device=x.device)[:, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, d))
        gate, value = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * value), (k, v)


class Crownless(nn.Module):
    def __init__(self, config=Config(), mode='copy'):
        super().__init__()
        if mode not in ('text', 'fields', 'copy'): raise ValueError('Unknown comparison arm')
        self.config, self.mode = config, mode
        self.embedding = nn.Embedding(config.vocab, config.dim)
        self.roles, self.knowledge = nn.Embedding(16, config.dim), nn.Embedding(4, config.dim)
        self.provenance, self.events = nn.Embedding(4, config.dim), nn.Embedding(4, config.dim)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.dim)
        self.copy_start = nn.Linear(config.dim, config.dim, bias=False)
        self.copy_end = nn.Linear(config.dim, config.dim, bias=False)
        self.copy_gate = nn.Linear(config.dim, 1)
        self.apply(self.initialize)
        frequencies = 10000 ** (-torch.arange(0, config.dim // config.heads, 2).float() / (config.dim // config.heads))
        angles = torch.arange(config.context).float()[:, None] * frequencies[None, :]
        self.register_buffer('cosine', angles.cos(), persistent=False)
        self.register_buffer('sine', angles.sin(), persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)

    def hidden(self, tokens, meta, caches=None):
        offset = 0 if caches is None else caches[0][0].shape[2]
        if offset + tokens.shape[1] > self.config.context: raise ValueError('Context exhausted')
        x = self.embedding(tokens)
        if self.mode != 'text':
            active = (meta[..., 0] != 0).unsqueeze(-1)
            x = x + active * (self.roles(meta[..., 0]) + self.knowledge(meta[..., 1]) +
                              self.provenance(meta[..., 2]) + self.events(meta[..., 3]))
        c, s = self.cosine[offset:offset + tokens.shape[1]], self.sine[offset:offset + tokens.shape[1]]
        saved = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, c, s, None if caches is None else caches[i])
            saved.append(cache)
        return self.norm(x), saved

    def heads(self, h, candidates, mask, source=None):
        source = h if source is None else source
        indices = candidates.unsqueeze(-1).expand(-1, -1, -1, self.config.dim)
        starts = source.gather(1, indices[:, :, 0])
        ends = source.gather(1, indices[:, :, 1])
        scores = (torch.einsum('btd,bkd->btk', self.copy_start(h), starts) +
                  torch.einsum('btd,bkd->btk', self.copy_end(h), ends)) / math.sqrt(self.config.dim)
        scores = scores.masked_fill(~mask[:, None], -1e4)
        return F.linear(h, self.embedding.weight), self.copy_gate(h).squeeze(-1), scores

    def loss(self, inputs):
        h, _ = self.hidden(inputs['tokens'], inputs['meta'])
        logits, gate, scores = self.heads(h, inputs['candidates'], inputs['candidate_mask'])
        if self.mode != 'copy':
            return F.cross_entropy(logits.flatten(0, 1), inputs['labels'].flatten(), ignore_index=-100)
        labels, target = inputs['copy_labels'], inputs['copy_targets']
        valid, copying = labels != -100, target >= 0
        lm = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction='none').view_as(labels)
        pointer = F.cross_entropy(scores.flatten(0, 1), target.clamp_min(0).flatten(), reduction='none').view_as(labels)
        selection = F.binary_cross_entropy_with_logits(gate, copying.float(), reduction='none')
        weights = inputs['weights'] * valid
        return ((selection + torch.where(copying, pointer, lm)) * weights).sum() / weights.sum()


@torch.no_grad()
def generate(model, tokenizer, record, device='cpu', max_tokens=160):
    model.eval()
    inputs = batch([record], device)
    n = record['prefix_length']
    h, caches = model.hidden(inputs['tokens'][:, :n], inputs['meta'][:, :n])
    source, current = h, h[:, -1:]
    generated, actions = [], []
    stopped = False
    for _ in range(max_tokens):
        logits, gate, scores = model.heads(current, inputs['candidates'], inputs['candidate_mask'], source)
        if model.mode == 'copy' and record['source_ids'] and gate.item() > 0:
            choice = scores[0, -1].argmax().item()
            tokens = record['source_ids'][choice]
            actions.append({'copy': choice})
        else:
            token = logits[0, -1].argmax().item()
            if token == tokenizer.token_to_id('[EOS]'):
                stopped = True
                break
            tokens = [token]
            actions.append({'token': token})
        if n + len(generated) + len(tokens) > model.config.context: break
        generated.extend(tokens)
        t = torch.tensor([tokens], device=device)
        current, caches = model.hidden(t, torch.zeros(1, len(tokens), 4, dtype=torch.long, device=device), caches)
        current = current[:, -1:]
    return {'text': tokenizer.decode(generated), 'stopped': stopped, 'actions': actions}


def save(path, model, tokenizer_path, extra):
    torch.save({'config': asdict(model.config), 'mode': model.mode,
                'tokenizer_sha256': hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
                'state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                **extra}, path)


def load(path, tokenizer_path, device='cpu'):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value['tokenizer_sha256'] != hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest():
        raise ValueError('Checkpoint tokenizer differs')
    model = Crownless(Config(**value['config']), value['mode'])
    model.load_state_dict(value['state'])
    return model.to(device), value
