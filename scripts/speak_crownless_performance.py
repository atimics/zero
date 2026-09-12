"""Speak a native held-account packet with explicit creature and emotion controls."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_performance import CREATURES, EMOTIONS, VERSION
from crownless_v2 import encode_row, generate
from crownless_v2_export import load_export
from speak_crownless_v2 import packet_record


def speak(model, metadata, tokenizer, packet, creature, emotion, history=()):
    if packet['grammar_sha256'] != metadata['rules_sha256']:
        raise ValueError('The model and held-account grammar differ')
    row = packet_record(packet, retold=packet.get('retellings', 0) >= 4)
    row['kind_id'] = metadata['meaning_ids'][packet['rule']]
    row['performance'] = {'creature':creature, 'emotion':emotion}
    row['history'] = list(history)
    result = generate(model, tokenizer, encode_row(tokenizer, row, slots=True, conversation=True))
    return {'performance':row['performance'], 'account':packet['text'],
            'confidence':packet['confidence'], 'history':row['history'], **result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--history', type=Path)
    p.add_argument('--creature', choices=CREATURES, required=True)
    p.add_argument('--emotion', choices=EMOTIONS, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    receipt = json.loads(args.model.with_name('performance.json').read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if receipt['schema'] != VERSION or receipt['model_sha256'] != digest:
        p.error('Use the model named in the performance receipt')
    model, metadata = load_export(args.model, args.tokenizer)
    result = speak(model, metadata, Tokenizer.from_file(str(args.tokenizer)),
        json.loads(args.packet.read_text()), args.creature, args.emotion,
        json.loads(args.history.read_text()) if args.history else ())
    print(json.dumps({'model_sha256':digest, **result}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
