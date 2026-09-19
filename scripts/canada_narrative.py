"""Prepare the accepted Canada delivery for matched ZERO experiments."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/canada-narrative-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(root, relative, expected):
    root = Path(root).resolve()
    file = (root / relative).resolve()
    if not file.is_relative_to(root) or digest(file) != expected:
        raise ValueError(f'Input identity mismatch: {relative}')
    return file


def encode_record(tokenizer, row):
    text = row['text']
    if not text.isascii() or hashlib.sha256(text.encode()).hexdigest() != row['contentHash']:
        raise ValueError('Record text identity mismatch')
    ids = []
    for line in re.findall(r'[^\n]*\n|[^\n]+$', text):
        encoded = tokenizer.encode(line).ids
        if tokenizer.decode(encoded) != line:
            raise ValueError('Tokenizer round trip mismatch')
        ids.extend(encoded)
    if len(ids) != row['metadata']['exactTokens']:
        raise ValueError('Record token count mismatch')
    return ids


def prepare(delivery, output):
    import numpy as np
    from tokenizers import Tokenizer
    lock = read_json(EXPERIMENT / 'inputs.lock.json')
    for name, sha in lock['delivery'].items():
        checked(delivery, name, sha)
    for item in lock['baseline'].values():
        checked(delivery, item['path'], item['sha256'])
    checked(delivery, lock['baseline_manifest']['path'], lock['baseline_manifest']['sha256'])
    for release in lock['releases'].values():
        for name, sha in release['files'].items():
            checked(delivery, release['directory'] + '/' + name, sha)
    output.mkdir(parents=True, exist_ok=False)
    for source, dest in [('train.bin', 'A.bin'), ('validation.bin', 'validation.bin'),
                         ('tokenizer.json', 'tokenizer.json'), ('token_bytes.json', 'token_bytes.json')]:
        shutil.copyfile(delivery / lock['baseline'][source]['path'], output / dest)
    tokenizer = Tokenizer.from_file(str(output / 'tokenizer.json'))
    counts = {'A': (output / 'A.bin').stat().st_size // 2}
    if counts['A'] != 27157194:
        raise ValueError('A control token count mismatch')
    membership = {}
    groups = {}
    for arm, release in lock['releases'].items():
        seen = {}; author_groups = set(); total = 0
        with (output / f'{arm}.bin').open('wb') as dst, (output / f'{arm}.records.jsonl').open('w') as index:
            with (delivery / release['directory'] / 'data/train.jsonl').open() as src:
                for line in src:
                    row = json.loads(line)
                    if row['split'] != 'train' or row['id'] in seen:
                        raise ValueError('Training record identity mismatch')
                    ids = encode_record(tokenizer, row)
                    np.asarray(ids, dtype='<u2').tofile(dst)
                    index.write(json.dumps({'id': row['id'], 'start': total, 'tokens': len(ids),
                                            'contentHash': row['contentHash']}) + '\n')
                    total += len(ids)
                    seen[row['id']] = row['contentHash']
                    author_groups.update(row['metadata']['authorGroupIds'])
        if total != release['tokens'] or len(seen) != release['records']:
            raise ValueError('Release counts mismatch')
        counts[arm] = total; membership[arm] = seen; groups[arm] = author_groups
        print(f'{arm}: {total} tokens, {len(seen)} records', flush=True)
    if any(membership['C'].get(key) != value for key, value in membership['B'].items()):
        raise ValueError('B/C membership mismatch')
    # The original A validation is protected by the accepted B/C overlap audit.
    # Split it again so checkpoint selection and the final score use separate targets.
    val = np.memmap(output / 'validation.bin', dtype='<u2', mode='r')
    midpoint = len(val) // 2
    selection = np.linspace(1024, midpoint - 256, 64, dtype=np.int64).tolist()
    outcome = np.linspace(midpoint + 1024, len(val) - 256, 256, dtype=np.int64).tolist()
    prompt_starts = np.linspace(midpoint + 1024, len(val) - 512, 200, dtype=np.int64).tolist()
    prompts = [{'id': i + 1, 'start': start,
                'prompt': tokenizer.decode(val[start:start + 128].tolist())}
               for i, start in enumerate(prompt_starts)]
    write_json(output / 'evaluation.json', {'selection': selection, 'outcome': outcome,
                                          'target_tokens_per_window': 256, 'prompts': prompts})
    manifest = {'schema': 1, 'status': 'prepared', 'training_presentations': 0,
                'lock_sha256': digest(EXPERIMENT / 'inputs.lock.json'), 'counts': counts,
                'evaluation_source': 'Frozen A validation; protected in accepted B/C delivery',
                'B_subset_C': True, 'files': {f.name: digest(f) for f in sorted(output.iterdir())}}
    write_json(output / 'manifest.json', manifest)
    return manifest


def verify_prepared(directory):
    manifest = read_json(directory / 'manifest.json')
    if manifest['lock_sha256'] != digest(EXPERIMENT / 'inputs.lock.json'):
        raise ValueError('Prepared input lock mismatch')
    for name, sha in manifest['files'].items():
        checked(directory, name, sha)
    required = {'A.bin', 'B.bin', 'C.bin', 'validation.bin', 'tokenizer.json',
                'token_bytes.json', 'evaluation.json', 'B.records.jsonl', 'C.records.jsonl'}
    if set(manifest['files']) != required:
        raise ValueError('Prepared file set mismatch')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.delivery, args.output)
