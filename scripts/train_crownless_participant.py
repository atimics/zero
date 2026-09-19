"""Train one-person JSON turns with the game's compact prefix and loss mask."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from crownless_v2 import Crownless, save
from crownless_v2_export import export, load_export


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_rows(path, tokenizer, smoke=False):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError('empty dataset')
    for row in rows:
        prompt = row['prompt']
        if prompt['format'] != 'crownless-person-v1':
            raise ValueError('unsupported participant format')
        status = row.get('source_review_status', '')
        if status.startswith('rejected') or status.startswith('hold'):
            raise ValueError('excluded source target')
        if not smoke and (row.get('review_status') != 'approved_compact' or
                          status != 'approved' or not row.get('world_group')):
            raise ValueError('training needs source and compact approval plus world_group')
        prefix = tokenizer.encode(prompt['text']).ids
        target = tokenizer.encode(row['target_text']).ids
        value = json.loads(row['target_text'])
        if not isinstance(value, dict):
            raise ValueError('target must be one object')
        speech = (set(value) == {'kind', 'text'} and value['kind'] == 'speech'
                  and isinstance(value['text'], str) and value['text'].strip()
                  and not any(ord(c) < 32 or ord(c) == 127 for c in value['text']))
        action = value == {'kind': 'action', 'action': 'end_conversation'}
        if not (speech or action):
            raise ValueError('target must be one speech or end-conversation action')
        if not prefix or len(prefix) > 352 or not target or len(target) >= 160:
            raise ValueError('native context budget exceeded')
        if len(row['target_text'].encode()) >= 512 or any(t < 9 for t in prefix + target):
            raise ValueError('native text boundary exceeded')
        if prefix != prompt['tokens'] or row['tokens'] != prefix + target:
            raise ValueError('tokenizer differs from compiled row')
        if row['labels'] != [-100] * (len(prefix) - 1) + target + [0]:
            raise ValueError('loss mask differs from actor-only shifted labels')
    return rows


def check_split(train, validation):
    if {r['world_group'] for r in train} & {r['world_group'] for r in validation}:
        raise ValueError('train and validation share a world history')
    def long_targets(rows):
        return {' '.join(r['target_text'].casefold().split()) for r in rows
                if len(r['target_text'].split()) >= 8}
    if long_targets(train) & long_targets(validation):
        raise ValueError('train and validation share full target text')


def loss(model, rows, device):
    length = max(len(r['tokens']) for r in rows)
    tokens = torch.zeros((len(rows), length), dtype=torch.long, device=device)
    labels = torch.full_like(tokens, -100)
    for i, row in enumerate(rows):
        n = len(row['tokens'])
        tokens[i, :n] = torch.tensor(row['tokens'], device=device)
        labels[i, :n] = torch.tensor(row['labels'], device=device)
    meta = torch.zeros((*tokens.shape, 16), dtype=torch.long, device=device)
    hidden, _ = model.hidden(tokens, meta)
    return F.cross_entropy(F.linear(hidden, model.embedding.weight).flatten(0, 1),
                           labels.flatten(), ignore_index=-100)


@torch.no_grad()
def sample(model, row, tokenizer):
    """Greedy JSON output with the native reserved-token and byte limits."""
    encoded = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet, extra = {}, 0
    for byte in range(256):
        char = chr(byte) if byte in encoded else chr(256 + extra)
        alphabet[char] = byte
        if byte not in encoded:
            extra += 1
    literal = {i: bytes(alphabet[c] for c in text) for text, i in tokenizer.get_vocab().items() if i >= 9}
    tokens = torch.tensor([row['prompt']['tokens']], dtype=torch.long)
    hidden, cache = model.hidden(tokens, torch.zeros((*tokens.shape, 16), dtype=torch.long))
    output, stopped = b'', False
    for _ in range(160):
        if cache[0][0].shape[2] >= 512:
            break
        logits = F.linear(hidden[:, -1], model.embedding.weight)[0]
        logits[1:9] = -torch.inf
        token = int(logits.argmax())
        if token == 0:
            stopped = True
            break
        if len(output) + len(literal[token]) >= 512:
            break
        output += literal[token]
        hidden, cache = model.hidden(torch.tensor([[token]]), torch.zeros((1, 1, 16), dtype=torch.long), cache)
    try:
        text = output.decode('utf-8')
        native_complete = stopped and bool(text) and all(ord(c) >= 32 and ord(c) != 127 for c in text)
    except UnicodeDecodeError:
        text, native_complete = output.decode('utf-8', errors='replace'), False
    return {'text': text, 'bytes_hex': output.hex(), 'eos': stopped, 'native_complete': native_complete}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--reference', type=Path, required=True, help='Reviewed native .ccv2 model')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialization', choices=['fresh', 'warm'], required=True)
    parser.add_argument('--smoke', action='store_true', help='Diagnostic run on pending examples; records limited scope')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if min(args.steps, args.batch_size) <= 0 or args.output.exists():
        parser.error('use positive counts and a fresh output directory')
    if not args.smoke and args.validation is None:
        parser.error('training requires held-out validation; use --smoke for a diagnostic')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    train = checked_rows(args.train, tokenizer, args.smoke)
    validation = checked_rows(args.validation, tokenizer) if args.validation else []
    if validation:
        check_split(train, validation)
    model, metadata = load_export(args.reference, args.tokenizer)
    dimensions = tuple(getattr(model.config, k) for k in ('dim', 'layers', 'heads', 'ff', 'vocab', 'context'))
    if dimensions != (192, 8, 6, 624, 4096, 512) or model.mode != 'conversation':
        raise ValueError('reference must match the native conversation architecture')
    if args.initialization == 'fresh':
        model = Crownless(model.config, model.mode)
    model.to(args.device)
    args.output.mkdir(parents=True)
    manifest = {'scope': 'pipeline_smoke' if args.smoke else 'held_out_world_training',
                'status': 'running', 'initialization': args.initialization,
                'parameters': sum(p.numel() for p in model.parameters()),
                'seed': args.seed, 'steps': args.steps, 'batch_size': args.batch_size,
                'device': args.device, 'torch': torch.__version__,
                'train_sha256': sha(args.train), 'validation_sha256': sha(args.validation) if validation else None,
                'reference_sha256': sha(args.reference), 'tokenizer_sha256': sha(args.tokenizer),
                'sources': {name: sha(Path(__file__).with_name(name)) for name in
                            ('train_crownless_participant.py', 'crownless_v2.py', 'crownless_v2_export.py')}}
    receipt = args.output / 'manifest.json'
    receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = random.Random(args.seed)
    started, target_tokens = time.monotonic(), 0
    try:
        with (args.output / 'history.jsonl').open('w') as history:
            for step in range(1, args.steps + 1):
                model.train()
                selected = [rng.choice(train) for _ in range(args.batch_size)]
                optimizer.zero_grad(set_to_none=True)
                value = loss(model, selected, args.device)
                if not torch.isfinite(value):
                    raise ValueError('non-finite training loss')
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                target_tokens += sum(sum(t != -100 for t in row['labels']) for row in selected)
                item = {'step': step, 'loss': value.item(), 'target_tokens': target_tokens,
                        'seconds': time.monotonic() - started}
                history.write(json.dumps(item) + '\n'); history.flush()
                if step == 1 or step == args.steps or step % 100 == 0:
                    print(json.dumps(item), flush=True)
        model.eval()
        if validation:
            with torch.no_grad():
                weights = [sum(t != -100 for t in r['labels']) for r in validation]
                manifest['validation_loss'] = sum(loss(model, [r], args.device).item() * n
                    for r, n in zip(validation, weights)) / sum(weights)
        model.cpu()
        save(args.output / 'last.pt', model, args.tokenizer,
             {'participant_format': 'crownless-person-v1', 'step': args.steps})
        export(model, args.tokenizer, args.output / 'last.ccv2', metadata)
        quantized, _ = load_export(args.output / 'last.ccv2', args.tokenizer)
        quantized.eval()
        samples = [{'source_line': r.get('source_line'), 'prefix': r['prompt']['text'],
                    'reference': r['target_text'], **sample(quantized, r, tokenizer)}
                   for r in (validation or train)[:2]]
        (args.output / 'samples.json').write_text(json.dumps(samples, indent=2) + '\n')
        manifest.update(status='complete', target_tokens=target_tokens,
                        seconds=time.monotonic()-started, export_sha256=sha(args.output/'last.ccv2'))
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), target_tokens=target_tokens)
        save(args.output / 'partial.pt', model, args.tokenizer,
             {'participant_format': 'crownless-person-v1', 'failed': True})
        raise
    finally:
        receipt.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
