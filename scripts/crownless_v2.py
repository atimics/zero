"""Crownless 5M v2: text, field, and source-copy controls on one backbone."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
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
    kinds: int = 137
    # Stance table sizes. Zero keeps the pre-typed architecture, so an export
    # written before this change still loads.
    voices: int = 0
    goals: int = 0
    levels: int = 0
    # Situation table rows per axis (hungry, sheltered, in_transit are binary).
    # Zero keeps the pre-situation architecture loadable the same way.
    situations: int = 0
    # Social tables: owes_listener, trusts_listener and far_from_home are
    # binary; faction is absent/crown/guild/commons. The count is the table
    # count (4); sizes are fixed. Zero when the run predates company.
    socials: int = 0


def train_tokenizer(rows, path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=['[EOS]'] + [f'[F{i}]' for i in range(8)],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tokenizer.train_from_iterator((r['prefix'] + r['output'] + '\n' for r in rows), trainer)
    tokenizer.save(str(path))
    return tokenizer


def field_marker(tokenizer, field):
    """The marker token for a copied span. Account fields use [F0]..[F7]. A
    recalled memory borrows [F7] as field 8: the marker budget is fixed, and a
    recall row's account carries few fields, so [F7] is otherwise free. This is
    a wire shared with the native runtime (token id 8)."""
    token = tokenizer.token_to_id('[F7]' if field >= 8 else f'[F{field}]')
    if token is None: raise ValueError('Tokenizer needs the field marker vocabulary')
    return token


def encode_parts(tokenizer, text, spans, slots=False):
    raw, tokens, mapped, at = text.encode(), [], [], 0
    for span in sorted(spans, key=lambda s: s['start']):
        start, end = span['start'], span['end']
        if not (at <= start <= end <= len(raw)): raise ValueError('Invalid or overlapping byte spans')
        if raw[start:end].decode() != span['text']: raise ValueError('Source span differs from its text')
        tokens.extend(tokenizer.encode(raw[at:start].decode()).ids)
        first = len(tokens)
        literal = tokenizer.encode(raw[start:end].decode()).ids
        if slots and span.get('spoken', span['role'] not in (0, 7, 8)) and span.get('knowledge', 0) != 3:
            tokens.append(field_marker(tokenizer, span['field']))
        else:
            tokens.extend(literal)
        mapped.append({**span, 'token_start': first, 'token_end': len(tokens), 'literal_ids': literal})
        at = end
    tokens.extend(tokenizer.encode(raw[at:].decode()).ids)
    if not slots and tokenizer.decode(tokens) != text: raise ValueError('Tokenizer round trip failed')
    return tokens, mapped


# Stance carried as typed ids rather than prompt text. Zero means absent, so an
# untyped row encodes exactly as it did before. The orders below are the C
# enums in cc_core_account.h offset by one; they are a wire format shared with
# the native runtime and cannot be reordered without regenerating the model.
VOICE_IDS = {'baker': 1, 'scribe': 2, 'farmer': 3, 'smith': 4, 'innkeeper': 5,
             'miller': 6, 'shepherd': 7, 'woodcutter': 8, 'resident': 9, 'quarryman': 10,
             'cartwright': 11, 'bandit': 12}
GOAL_IDS = {'keep_order': 1, 'secure_livelihood': 2, 'survive_crisis': 3, 'carry_news': 4}
LEVEL_IDS = {'low': 1, 'medium': 2, 'high': 3}
META_FIELDS = 16
# An unremarkable situation: fed, sheltered, home. Plain rows and the native
# runtime's default mind agree on it, so neither side invents a predicament.
PLAIN_SITUATION = {'hungry': False, 'sheltered': True, 'in_transit': False}
SITUATION_AXES = ('hungry', 'sheltered', 'in_transit')
# Nobody owed, nobody trusted especially, no faction, home. Same agreement.
PLAIN_SOCIAL = {'owes_listener': False, 'trusts_listener': False, 'faction': None,
                'far_from_home': False}
SOCIAL_AXES = ('owes_listener', 'trusts_listener', 'faction', 'far_from_home')
FACTION_IDS = {'crown': 1, 'guild': 2, 'commons': 3}


def typed_stance_for(model):
    """True when this checkpoint reads stance from the meta channel.

    Checkpoints with voice tables ignore stance text; older ones have no
    tables and read the # voice / # goal / # stress / # courage lines
    instead. Callers pass this to encode_row so one script serves both
    generations without a flag to get wrong.
    """
    return bool(getattr(getattr(model, 'config', None), 'voices', 0))


def channels_for(model):
    """Every conditioning channel this checkpoint reads, as encode_row kwargs.

    One script serves all generations this way: untyped checkpoints keep
    byte-identical behavior (all False) and newer ones get every meta id they
    were trained on, with no flag to get wrong.
    """
    config = getattr(model, 'config', None)
    return {'typed_stance': bool(getattr(config, 'voices', 0)),
            'situation': bool(getattr(config, 'situations', 0)),
            'social': bool(getattr(config, 'socials', 0))}


def encode_row(tokenizer, row, context=512, slots=False, packet=False, conversation=False,
               typed_stance=False, situation=False, social=False):
    prefix, fields = encode_parts(tokenizer, row['prefix'], row['fields'], slots)
    if packet or conversation:
        # The runtime's CcCoreModelBegin sends this stance when the caller
        # names no character, so a row without one encodes the same way.
        mind = row.get('mind') or {'goal': 'secure_livelihood', 'stress': 'medium',
                                   'courage': 'medium'}
        # The witnessed cue and the trailing control line belong to the mind
        # context. Rows without one keep the plain account shape the native
        # runtime emits from CcCoreModelBegin.
        cue = '- ' + ('? ' if row.get('confidence', 80) < 40 else '') + ('~ ' if row.get('retold') else '') + \
            ('! ' if mind and row.get('witnessed') else '')
        prefix = []
        if conversation:
            # Mind context lines precede the spoken history: goal, stress,
            # courage, memories, and current thoughts. The model reads them as
            # plain text; only the held account's fields carry markers.
            mind_lines = []
            if not typed_stance:
                mind_lines.append('# voice: ' + (row.get('voice') or 'resident'))
                if mind.get('goal'): mind_lines.append('# goal: ' + mind['goal'])
                if mind.get('stress'): mind_lines.append('# stress: ' + mind['stress'])
                if mind.get('courage'): mind_lines.append('# courage: ' + mind['courage'])
            for memory in mind.get('memories', [])[-2:]:
                mind_lines.append('# memory: ' + memory)
            for thought in mind.get('thoughts', [])[-2:]:
                mind_lines.append('# thought: ' + thought)
            prefix = [token for line in mind_lines for token in tokenizer.encode(line + '\n').ids]
            # Keep literal speech, with exact mentions of the avatar's known
            # fields represented by their existing markers. The model learns
            # responses from these words; dialogue acts are training labels only.
            known = {f['text']: f['field'] for f in fields if f.get('spoken') and f['knowledge'] != 3}
            pattern = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in sorted(known, key=len, reverse=True)) + r')(?!\w)') if known else None
            messages = []
            for message in row.get('history', [])[-4:]:
                if message['speaker'] not in ('self', 'other'): raise ValueError('Invalid speaker')
                speech = message['text']
                if pattern: speech = pattern.sub(lambda m: f'[F{known[m.group()]}]', speech)
                part = tokenizer.encode(message['speaker'] + ': ' + speech + '\n').ids
                if len(part) > 256: raise ValueError('A spoken event exceeds the history budget')
                messages.append(part)
            while sum(map(len, messages)) > 256: messages.pop(0)
            prefix.extend(token for message in messages for token in message)
            if 'performance' in row:
                from crownless_performance import control_text
                prefix.extend(tokenizer.encode(control_text(row['performance'])).ids)
        prefix.extend(tokenizer.encode(cue).ids)
        selected = []
        for field in sorted(fields, key=lambda f: f['field']):
            if not field.get('spoken', field['role'] not in (0, 7, 8)): continue
            start = len(prefix)
            prefix.append(field_marker(tokenizer, field['field']))
            selected.append({**field, 'token_start': start, 'token_end': len(prefix), 'event': 1})
        # A recalled memory is offered as a copy candidate beside the account
        # fields: the reply reproduces it exactly instead of reaching for a
        # memorised line. Field 8 borrows marker [F7]; only recall rows add it.
        if conversation and row.get('control') == 'recall' and mind.get('memories'):
            memory = mind['memories'][-1]
            start = len(prefix)
            prefix.append(field_marker(tokenizer, 8))
            selected.append({'field': 8, 'role': 0, 'knowledge': 0, 'provenance': 3,
                             'spoken': True, 'event': 1, 'token_start': start,
                             'token_end': len(prefix), 'literal_ids': tokenizer.encode(memory).ids})
        prefix.extend(tokenizer.encode('\n').ids)
        fields = selected
        # A mind context ends with a control cue that names the output:
        # say: for spoken lines, # think: for internal thoughts.
        if mind:
            prefix.extend(tokenizer.encode('# ' + row.get('control', 'say') + ':\n').ids)
    output, copies = encode_parts(tokenizer, row['output'], row['copies'], slots)
    tokens = prefix + output + [tokenizer.token_to_id('[EOS]')]
    if len(tokens) > context + 1: raise ValueError(f"Example exceeds context: {row['id']}")
    n = len(tokens) - 1
    meta = [[0, 0, 0, 0, row.get('kind_id', 0) if i < len(prefix) else 0] + [0] * 11
            for i in range(n)]
    if typed_stance and (packet or conversation):
        # Added at every prefix position rather than retrieved by attention from
        # one line forty tokens back, which is the whole point of the change.
        stance = [VOICE_IDS.get(row.get('voice') or 'resident', 0),
                  GOAL_IDS.get(mind.get('goal', ''), 0),
                  LEVEL_IDS.get(mind.get('stress', ''), 0),
                  LEVEL_IDS.get(mind.get('courage', ''), 0)]
        for i in range(min(len(prefix), n)):
            meta[i][5:9] = stance
    if situation and (packet or conversation):
        # Binary axes, always present: 1 is the state, 0 the learned "not so".
        # No gate, unlike stance - there is no absent situation to skip for.
        state = row.get('situation') or PLAIN_SITUATION
        ids = [1 if state.get(axis, PLAIN_SITUATION[axis]) else 0 for axis in SITUATION_AXES]
        for i in range(min(len(prefix), n)):
            meta[i][9:12] = ids
    if social and (packet or conversation):
        # Same bargain one level up: debts, trust and distance are always
        # known; faction may be absent (0, gated like voice).
        company = row.get('social') or PLAIN_SOCIAL
        ids = [1 if company.get('owes_listener', False) else 0,
               1 if company.get('trusts_listener', False) else 0,
               FACTION_IDS.get(company.get('faction') or '', 0),
               1 if company.get('far_from_home', False) else 0]
        for i in range(min(len(prefix), n)):
            meta[i][12:16] = ids
    candidates, source_ids, feedback_ids, field_to_candidate = [], [], [], {}
    for field in fields:
        for i in range(field['token_start'], field['token_end']):
            meta[i][:4] = [field['role'], field['knowledge'], field['provenance'], field['event']]
        if field.get('spoken', field['role'] not in (0, 7, 8)) and field['knowledge'] != 3:
            field_to_candidate[field['field']] = len(candidates)
            candidates.append([field['token_start'], field['token_end'] - 1])
            source_ids.append(field['literal_ids'])
            feedback_ids.append(prefix[field['token_start']:field['token_end']])
    labels = [-100] * (len(prefix) - 1) + tokens[len(prefix):]
    copy_targets, weights = [-1] * n, [1.] * n
    copy_labels = list(labels)
    for copied in copies:
        start = len(prefix) + copied['token_start'] - 1
        end = len(prefix) + copied['token_end'] - 1
        slot = field_to_candidate[copied['field']]
        if output[copied['token_start']:copied['token_end']] != feedback_ids[slot]:
            raise ValueError('Copy token sequence differs from source')
        copy_targets[start], weights[start] = slot, float(len(source_ids[slot]))
        for i in range(start + 1, end): copy_labels[i] = -100
    return {'tokens': tokens[:-1], 'meta': meta, 'labels': labels, 'copy_labels': copy_labels,
            'copy_targets': copy_targets, 'weights': weights, 'candidates': candidates,
            'source_ids': source_ids, 'feedback_ids': feedback_ids,
            'prefix_length': len(prefix), 'row': row}


def batch(records, device):
    length = max(len(r['tokens']) for r in records)
    slots = max(1, max(len(r['candidates']) for r in records))
    b = len(records)
    out = {'tokens': torch.zeros(b, length, dtype=torch.long),
           'meta': torch.zeros(b, length, META_FIELDS, dtype=torch.long),
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
        if mode not in ('text', 'fields', 'copy', 'slots', 'packet', 'conversation'): raise ValueError('Unknown comparison arm')
        self.config, self.mode = config, mode
        self.embedding = nn.Embedding(config.vocab, config.dim)
        self.roles, self.knowledge = nn.Embedding(16, config.dim), nn.Embedding(4, config.dim)
        self.provenance, self.events = nn.Embedding(4, config.dim), nn.Embedding(4, config.dim)
        self.kinds = nn.Embedding(config.kinds, config.dim) if config.kinds else None
        # A labelled line for the stance: added into the residual stream at
        # every position instead of competing for attention as prompt text.
        self.voices = nn.Embedding(config.voices, config.dim) if config.voices else None
        self.goals = nn.Embedding(config.goals, config.dim) if config.voices else None
        self.stresses = nn.Embedding(config.levels, config.dim) if config.voices else None
        self.courages = nn.Embedding(config.levels, config.dim) if config.voices else None
        # The situation rides the same labelled line: three binary axes, always
        # present, so no gate. Index 0 is the learned "not so" vector, not absence.
        self.hungry = nn.Embedding(2, config.dim) if config.situations else None
        self.sheltered = nn.Embedding(2, config.dim) if config.situations else None
        self.in_transit = nn.Embedding(2, config.dim) if config.situations else None
        # Company: two binaries always present, faction categorical with 0
        # absent (gated like voice), far_from_home binary.
        self.owes = nn.Embedding(2, config.dim) if config.socials else None
        self.trusts = nn.Embedding(2, config.dim) if config.socials else None
        self.faction = nn.Embedding(4, config.dim) if config.socials else None
        self.far = nn.Embedding(2, config.dim) if config.socials else None
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
            if self.kinds is not None:
                x = x + (meta[..., 4] != 0).unsqueeze(-1) * self.kinds(meta[..., 4])
            if self.voices is not None and meta.shape[-1] > 5:
                x = x + (meta[..., 5] != 0).unsqueeze(-1) * (
                    self.voices(meta[..., 5]) + self.goals(meta[..., 6]) +
                    self.stresses(meta[..., 7]) + self.courages(meta[..., 8]))
            if self.hungry is not None and meta.shape[-1] > 9:
                x = x + self.hungry(meta[..., 9]) + self.sheltered(meta[..., 10]) + \
                    self.in_transit(meta[..., 11])
            if self.owes is not None and meta.shape[-1] > 12:
                x = x + self.owes(meta[..., 12]) + self.trusts(meta[..., 13]) + \
                    self.far(meta[..., 15]) + (meta[..., 14] != 0).unsqueeze(-1) * \
                    self.faction(meta[..., 14])
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
        if self.mode not in ('copy', 'slots', 'packet', 'conversation'):
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
    used = n
    for _ in range(max_tokens):
        logits, gate, scores = model.heads(current, inputs['candidates'], inputs['candidate_mask'], source)
        if model.mode in ('copy', 'slots', 'packet', 'conversation') and record['source_ids'] and gate.item() > 0:
            choice = scores[0, -1].argmax().item()
            tokens = record['source_ids'][choice]
            feedback = record['feedback_ids'][choice]
            actions.append({'copy': choice})
        else:
            for i in range(8):
                marker = tokenizer.token_to_id(f'[F{i}]')
                if marker is not None: logits[0, -1, marker] = -torch.inf
            token = logits[0, -1].argmax().item()
            if token == tokenizer.token_to_id('[EOS]'):
                stopped = True
                break
            tokens = [token]
            feedback = tokens
            actions.append({'token': token})
        if used + len(feedback) > model.config.context: break
        generated.extend(tokens)
        used += len(feedback)
        t = torch.tensor([feedback], device=device)
        current, caches = model.hidden(t, torch.zeros(1, len(feedback), META_FIELDS, dtype=torch.long, device=device), caches)
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
    model = Crownless(Config(**{'kinds': 0, **value['config']}), value['mode'])
    model.load_state_dict(value['state'])
    return model.to(device), value
