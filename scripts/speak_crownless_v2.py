"""Speak from the game's native held-account packet using the v2 core."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from tokenizers import Tokenizer
from crownless_v2 import encode_row, generate, load
from crownless_v2_export import load_export


def packet_record(packet, context=(), retold=False):
    earlier = ''.join('- ' + event + '\n' for event in context)
    cue = ('? ' if packet['confidence'] < 40 else '') + ('~ ' if retold else '')
    opening = earlier + '- ' + cue
    fields = [{**f, 'start': f['start'] + len(opening.encode()),
               'end': f['end'] + len(opening.encode()), 'event': 2 if context else 1}
              for f in packet['fields']]
    return {'id': 'live', 'kind_id': packet['kind'] + 1, 'prefix': opening + packet['text'] + '\n',
            'confidence': packet['confidence'], 'retold': retold,
            'output': '', 'fields': fields, 'copies': []}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--account-binary', type=Path, required=True)
    p.add_argument('--kind', type=int, required=True)
    p.add_argument('--event', action='append', required=True)
    p.add_argument('--confidence', type=int, default=80)
    p.add_argument('--retold', action='store_true')
    p.add_argument('--trace', type=Path)
    args = p.parse_args()
    if not 1 <= len(args.event) <= 3: p.error('Supply one to three events, with the topic last')
    torch.set_num_threads(4)
    packet = json.loads(subprocess.check_output([str(args.account_binary), str(args.kind),
                         str(args.confidence), '0', args.event[-1], '--packet'], text=True))
    model, metadata = (load_export if args.model.suffix == '.ccv2' else load)(args.model, args.tokenizer)
    if metadata.get('rules_sha256') != packet['grammar_sha256']:
        p.error('The model and game account grammar differ')
    if model.mode != 'packet' and model.config.kinds and packet['kind'] + 1 >= model.config.kinds:
        p.error('This event kind needs a newer model')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    row = packet_record(packet, args.event[:-1], args.retold)
    if model.mode == 'packet':
        if packet['rule'] not in metadata['meaning_ids']: p.error('This account meaning needs a newer model')
        row['kind_id'] = metadata['meaning_ids'][packet['rule']]
    record = encode_row(tokenizer, row, model.config.context, slots=model.mode in ('slots', 'packet'), packet=model.mode == 'packet')
    result = generate(model, tokenizer, record)
    if args.trace: args.trace.write_text(json.dumps({'packet': packet, **result}, indent=2) + '\n')
    print(result['text'])
