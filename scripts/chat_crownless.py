"""Alternate two avatars, appending every generated sentence as spoken history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from tokenizers import Tokenizer
from crownless_v2 import encode_row, generate, load
from crownless_v2_export import load_export
from speak_crownless_v2 import packet_record


def chat(model, metadata, tokenizer, avatars, packets, turns=6, first=None):
    if model.mode != 'conversation': raise ValueError('Use a conversation checkpoint')
    history = [] if first is None else [{'speaker': 1, 'text': first}]
    result = []
    for turn in range(turns):
        speaker = turn % 2
        avatar, packet = avatars[speaker], packets[speaker]
        row = packet_record(packet, retold=avatar.get('retellings', 0) >= 4)
        row['kind_id'] = metadata['meaning_ids'][packet['rule']]
        row['history'] = [{'speaker': 'self' if h['speaker'] == speaker else 'other', 'text': h['text']}
                          for h in history[-4:]]
        record = encode_row(tokenizer, row, slots=True, conversation=True)
        generated = generate(model, tokenizer, record)
        result.append({'turn': turn + 1, 'speaker': avatar['name'], 'account': packet['text'],
                       'confidence': packet['confidence'], 'history': row['history'], **generated,
                       'input_sha256': hashlib.sha256(json.dumps({k: record[k] for k in
                           ('tokens', 'meta', 'candidates', 'source_ids')}, sort_keys=True).encode()).hexdigest()})
        history.append({'speaker': speaker, 'text': generated['text']})
        if not generated['stopped']: break
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokenizer', 'avatars', 'account-binary'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--turns', type=int, default=6)
    p.add_argument('--first')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not 1 <= args.turns <= 32: p.error('Use one to 32 turns')
    torch.set_num_threads(4)
    model, metadata = (load_export if args.model.suffix == '.ccv2' else load)(args.model, args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    avatars = json.loads(args.avatars.read_text())
    if len(avatars) != 2: p.error('Supply exactly two avatars')
    packets = []
    for avatar in avatars:
        packet = json.loads(subprocess.check_output([str(args.account_binary), str(avatar['kind']),
                            str(avatar['confidence']), '0', avatar['account'], '--packet'], text=True))
        if metadata['rules_sha256'] != packet['grammar_sha256']: p.error('The model and native grammar differ')
        packets.append(packet)
    turns = chat(model, metadata, tokenizer, avatars, packets, args.turns, args.first)
    for row in turns: print(row['speaker'] + ': ' + row['text'])
    if args.output:
        args.output.write_text(json.dumps({'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'avatars': avatars, 'turns': turns}, indent=2) + '\n')


if __name__ == '__main__': main()
