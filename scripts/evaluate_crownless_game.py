"""Check a saved core against distinct held accounts from a native game run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from tokenizers import Tokenizer
from crownless_v2 import encode_row, generate, load
from crownless_v2_export import load_export
from score_crownless_v2 import accepted_forms
from speak_crownless_v2 import packet_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'tokenizer', 'account-binary', 'game-corpus', 'rules', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, metadata = (load_export if args.model.suffix == '.ccv2' else load)(args.model, args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rules = {r['id']: r for r in json.loads(args.rules.read_text())['rules']}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(args.rules) != metadata['rules_sha256']: p.error('Model and rules differ')
    rows, seen, observations = [], set(), 0
    for line in args.game_corpus.read_text().splitlines():
        source = json.loads(line)
        observations += 1
        account = source['input']
        kind = int(source['rule'].split(':')[0])
        retold = account['retellings'] >= 4
        # The encoder uses these confidence bands; keep the first provenance.
        key = kind, account['account'], account['confidence'] < 40, retold
        if key in seen: continue
        seen.add(key)
        probe = subprocess.run([str(args.account_binary), str(kind), str(account['confidence']),
                                '0', account['account'], '--packet'], capture_output=True, text=True)
        item = {'account': account, 'provenance': source['provenance'], 'kind': kind}
        if probe.returncode:
            rows.append({**item, 'parsed': False, 'approved_meaning': False})
            continue
        packet = json.loads(probe.stdout)
        if packet['grammar_sha256'] != metadata['rules_sha256']: p.error('Native grammar differs')
        row = packet_record(packet, retold=retold)
        row['kind_id'] = metadata['meaning_ids'][packet['rule']] if model.mode == 'packet' else kind + 1
        record = encode_row(tokenizer, row, model.config.context,
                            slots=model.mode in ('slots', 'packet'), packet=model.mode == 'packet')
        started = time.perf_counter()
        result = generate(model, tokenizer, record)
        elapsed = time.perf_counter() - started
        rows.append({**item, 'parsed': True, 'rule': packet['rule'], **result,
                     'seconds': elapsed,
                     'approved_meaning': result['stopped'] and result['text'] in accepted_forms(row, rules[packet['rule']])})
    report = {'scope': 'Distinct native held accounts and confidence bands from one supplied game corpus.',
              'identities': {name: digest(getattr(args, name)) for name in
                             ('model', 'tokenizer', 'rules', 'game_corpus', 'account_binary')},
              'observations': observations, 'distinct_accounts': len(rows),
              'parsed': sum(r['parsed'] for r in rows),
              'approved_meaning': sum(r['approved_meaning'] for r in rows),
              'kinds': sorted({r['kind'] for r in rows}), 'rows': rows}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'identities')}))


if __name__ == '__main__': main()
