"""Continue the Crownless 5M on a fixed Crownless/external retrieval mix."""
import argparse
import hashlib
import json
import random
import re
import struct
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from crownless_v2 import batch, encode_row, generate
from crownless_v2_export import export, load_export
from train_crownless_conversation import build_rows, read


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def braid_tokenizer_decode(tokenizer_path, ids):
    data = json.loads(Path(tokenizer_path).read_text())
    vocab = {int(index): token for token, index in data['model']['vocab'].items()}
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    inverse = {}
    extra = 256
    for value in range(256):
        codepoint = value if value in direct else extra
        inverse[chr(codepoint)] = value
        if value not in direct: extra += 1
    raw = bytearray()
    for token_id in ids:
        token = vocab.get(int(token_id))
        if token is None: raise ValueError(f'Unknown Braid token {token_id}')
        for char in token:
            if char not in inverse: raise ValueError('Braid token is not byte-decodable')
            raw.append(inverse[char])
    return raw.decode('ascii', errors='strict')


def braid_pack_text(pack_path, braid_tokenizer_path):
    raw = Path(pack_path).read_bytes()
    if raw[:8] != b'BRAIDPK1': raise ValueError(f'Invalid Braid pack: {pack_path}')
    metadata_length = struct.unpack_from('<I', raw, 16)[0]
    token_count = struct.unpack_from('<I', raw, 20)[0]
    width = struct.unpack_from('<I', raw, 24)[0]
    metadata = json.loads(raw[64:64 + metadata_length])
    offset = 64 + metadata_length
    ids = [int.from_bytes(raw[offset + i * width:offset + (i + 1) * width], 'little')
           for i in range(metadata['activeTokens'])]
    return braid_tokenizer_decode(braid_tokenizer_path, ids)


def neural_embedding(text, encoder):
    return encoder.encode(text, normalize_embeddings=True).tolist()


def cosine(left, right): return sum(a * b for a, b in zip(left, right))


def write_semantic_index(output, entries, vectors, manifest, embedder_sha, embedding_id):
    import hashlib as _hashlib
    output.mkdir(parents=True, exist_ok=True)
    entries_bytes = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode()
    (output / 'semantic-entries.json').write_bytes(entries_bytes)
    raw = bytearray()
    for vector in vectors:
        raw.extend(struct.pack('<f', value) for value in vector) if False else None
    raw = b''.join(struct.pack('<f', value) for vector in vectors for value in vector)
    (output / 'semantic-vectors.bin').write_bytes(raw)
    vectors_digest = _hashlib.sha256(bytes(raw)).hexdigest()
    index_id = 'braid_semantic_index_' + _hashlib.sha256(entries_bytes + bytes(raw)).hexdigest()
    index = {'schemaVersion': 'braid.semantic-retrieval-index/v1', 'indexId': index_id,
             'streamId': f'braid_stream_{manifest["manifestDigest"]}',
             'manifestDigest': manifest['manifestDigest'], 'embeddingId': embedding_id,
             'embeddingSha256': embedder_sha, 'dimension': len(vectors[0]), 'metric': 'cosine',
             'normalized': True, 'entryCount': len(entries),
             'entriesSha256': _hashlib.sha256(entries_bytes).hexdigest(),
             'vectorsSha256': vectors_digest}
    (output / 'semantic-index.json').write_text(json.dumps(index, indent=2) + '\n')
    return index


def query_external(texts, embeddings, query, encoder, count):
    vector = neural_embedding(query, encoder)
    if vector is None: return texts[:count]
    terms = {word.lower() for word in query.split() if len(word) >= 4}
    def score(index):
        words = {word.lower().strip('.,;:!?()[]') for word in texts[index].split() if len(word) >= 4}
        overlap = len(terms & words) / max(1, len(terms))
        return cosine(vector, embeddings[index]) + 0.2 * overlap
    ranked = sorted(range(len(texts)), key=lambda i: (-score(i), i))
    return [texts[i] for i in ranked[:count]]


def load_external(args, embedder=None, embedder_tokenizer=None):
    manifest = json.loads(Path(args.braid_manifest).read_text())
    pack_dir = Path(args.braid_manifest).parent / 'packs'
    tokenizer = Path(args.braid_tokenizer)
    texts = []
    for descriptor in manifest['packs'][:args.packs]:
        pack = next(pack_dir.glob(f'{descriptor["sequenceIndex"]:08d}-*.braidpack'))
        text = braid_pack_text(pack, tokenizer)
        if text.strip(): texts.append(text)
    if not texts: raise ValueError('Braid stream produced no text')
    return texts, [], manifest


def style_candidates(texts):
    candidates = []
    for text in texts:
        for sentence in re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')):
            sentence = ' '.join(sentence.split()).strip()
            if not 35 <= len(sentence) <= 180 or any(char.isdigit() for char in sentence): continue
            words = sentence.split()
            capitals = sum(word[:1].isupper() for word in words[1:])
            if capitals > 1 or sentence.startswith(('Chapter ', 'THE ', 'PART ')): continue
            if sentence.count(';') > 1 or sentence.count(':') > 1: continue
            candidates.append(sentence)
    return sorted(set(candidates))


def style_record(tokenizer, text, context=512):
    return external_record(tokenizer, '# style:\n' + text, context)


def external_record(tokenizer, text, context=512):
    ids = tokenizer.encode(text).ids + [tokenizer.token_to_id('[EOS]')]
    ids = ids[:context + 1]
    if len(ids) < 2: raise ValueError('External text chunk is too short')
    n = len(ids) - 1
    return {'tokens': ids[:-1], 'meta': [[0, 0, 0, 0, 0]] * n,
            'labels': ids[1:], 'copy_labels': ids[1:], 'copy_targets': [-1] * n,
            'weights': [1.0] * n, 'candidates': [[0, 0]], 'source_ids': [],
            'feedback_ids': [], 'prefix_length': 0,
            'row': {'source': 'braid', 'text': text}}




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--braid-manifest', type=Path, required=True)
    parser.add_argument('--braid-tokenizer', type=Path, required=True)
    parser.add_argument('--packs', type=int, default=512)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--external-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=91)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a fresh output directory')
    torch.manual_seed(args.seed); random.seed(args.seed); torch.set_num_threads(4)
    model, metadata = load_export(args.base, args.tokenizer, args.device)
    teacher, _ = load_export(args.base, args.tokenizer, args.device)
    teacher.eval()
    model.mode = 'conversation'
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads((args.data / 'rules.json').read_text())['rules']}
    bases = {split: read(args.data / f'{split}.jsonl') for split in ('train', 'validation', 'test')}
    for rows in bases.values():
        for row in rows: row['kind_id'] = metadata['meaning_ids'][row['rule']]
    crownless = build_rows(bases['train'], rules, f'{args.seed}:train', repeats=2)
    external_texts_all, _, braid_manifest = load_external(args)
    encoder = SentenceTransformer(args.embedding_model, device='mps' if args.device == 'mps' else 'cpu')
    pairs = [(text, neural_embedding(text, encoder)) for text in external_texts_all]
    external_texts = [text for text, vector in pairs if vector is not None]
    external_vectors = [vector for text, vector in pairs if vector is not None]
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    external = [external_record(tokenizer, text) for text in external_texts]
    style_texts = style_candidates(external_texts)
    style_records = [style_record(tokenizer, text) for text in style_texts]
    external_by_text = {record['row']['text']: record for record in external}
    entries = [{'entryId': f'narrative-{index}', 'recordId': f'pack-{index}', 'textSha256': hashlib.sha256(text.encode()).hexdigest(), 'vectorOrdinal': index}
               for index, text in enumerate(external_texts)]
    write_semantic_index(output, entries, external_vectors, braid_manifest, hashlib.sha256(args.embedding_model.encode()).hexdigest(), args.embedding_model)
    training = [encode_row(tokenizer, row, slots=True, conversation=True) for row in crownless]
    grounding_rows = []
    for row in bases['train'][:max(1000, len(bases['train']) // 10)]:
        grounded = dict(row)
        grounded['history'] = []
        grounded['control'] = 'say'
        grounded['act'] = 'grounding'
        grounding_rows.append(encode_row(tokenizer, grounded, slots=True, conversation=True))
    validation = [encode_row(tokenizer, row, slots=True, conversation=True)
                  for row in build_rows(bases['validation'], rules, f'{args.seed}:validation')[:244]]
    (output / 'manifest.json').write_text(json.dumps({
        'base_sha256': sha(args.base), 'tokenizer_sha256': sha(args.tokenizer),
        'embedding': args.embedding_model, 'braid_manifest_digest': braid_manifest['manifestDigest'],
        'braid_packs': len(external), 'style_candidates': len(style_records), 'external_ratio': args.external_ratio,
        'crownless_rows': len(training), 'seed': args.seed, 'steps': args.steps,
    }, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    rng = random.Random(args.seed); best = float('inf'); history = []
    batch_size = 20
    style_count = max(1, round(batch_size * args.external_ratio))
    grounding_count = 3
    for step in range(1, args.steps + 1):
        model.train()
        crown_records = rng.choices(training, k=batch_size - style_count - grounding_count)
        grounding_batch = rng.choices(grounding_rows, k=grounding_count)
        style_batch = rng.choices(style_records, k=style_count)
        records = crown_records + grounding_batch + style_batch
        inputs = batch(records, args.device)
        crown_inputs = batch(crown_records + grounding_batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs)
        student_hidden, _ = model.hidden(crown_inputs['tokens'], crown_inputs['meta'])
        teacher_hidden, _ = teacher.hidden(crown_inputs['tokens'], crown_inputs['meta'])
        student_logits, _, _ = model.heads(student_hidden, crown_inputs['candidates'], crown_inputs['candidate_mask'])
        with torch.no_grad():
            teacher_logits, _, _ = teacher.heads(teacher_hidden, crown_inputs['candidates'], crown_inputs['candidate_mask'])
        distill = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction='batchmean')
        loss = loss + 0.1 * distill
        if not torch.isfinite(loss): raise ValueError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                values = [model.loss(batch(validation[i:i + 16], args.device)).item()
                          for i in range(0, len(validation), 16)]
            val = sum(values) / len(values)
            item = {'step': step, 'loss': loss.item(), 'crownless_validation': val, 'distill': distill.item(), 'style_ratio': style_count / batch_size}
            history.append(item); (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(item), flush=True)
            if val < best:
                best = val
                torch.save({'state': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'step': step}, output / 'best.pt')
    torch.save({'state': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'step': args.steps}, output / 'final.pt')
    saved = torch.load(output / 'final.pt', map_location='cpu', weights_only=True)
    model.to('cpu').load_state_dict(saved['state'])
    export(model, args.tokenizer, output / 'core.ccv2', metadata | {'step': saved['step'], 'seed': args.seed})
    print(json.dumps({'model_sha256': sha(output / 'core.ccv2'), 'best_step': saved['step'], 'external_chunks': len(external)}))


if __name__ == '__main__': main()
