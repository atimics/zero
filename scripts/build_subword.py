"""Fit a byte-level BPE on training text and freeze reversible token streams."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train([str(source / 'train.txt')], trainers.BpeTrainer(
        vocab_size=2048, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tokenizer.save(str(output / 'tokenizer.json'))
    manifest = {'vocabulary': tokenizer.get_vocab_size(), 'training_split_only': True,
                'encoding': 'Each line includes its original newline; byte-level BPE with no added prefix.', 'splits': {}}
    for split in ['train', 'validation', 'test']:
        count = 0
        with (output / f'{split}.bin').open('wb') as dst:
            with (source / f'{split}.txt').open(newline='') as src:
                for line in src:
                    ids = tokenizer.encode(line).ids
                    if tokenizer.decode(ids) != line:
                        raise ValueError('Tokenizer round trip failed')
                    np.asarray(ids, dtype='<u2').tofile(dst)
                    count += len(ids)
        raw = (source / f'{split}.txt').read_bytes()
        manifest['splits'][split] = {'bytes':len(raw), 'tokens':count, 'source_sha256':hashlib.sha256(raw).hexdigest()}
    lengths = [len(tokenizer.decode([i]).encode()) for i in range(tokenizer.get_vocab_size())]
    for split in ['train', 'validation', 'test']:
        ids = np.fromfile(output / f'{split}.bin', dtype='<u2')
        if int(np.asarray(lengths)[ids].sum()) != manifest['splits'][split]['bytes']:
            raise ValueError('Token byte accounting differs from source')
    (output / 'token_bytes.json').write_text(json.dumps(lengths))
    manifest['files'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
