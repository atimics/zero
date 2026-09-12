"""Score complete approved meanings separately from name presence and exact targets."""
import argparse
import hashlib
import json
from pathlib import Path


def accepted_forms(row, rule):
    values = [''] * len(rule['roles'])
    for field in row['fields']: values[field['field']] = field['text']
    endings = [', if the story is right.', ', if the rumour is true.'] if row['confidence'] < 40 else (
        [', so people say.', ', according to the word going round.'] if row['retold'] else ['.'])
    forms = set()
    for template in rule['outputs']:
        text = template.format(*values)
        text = text[0].upper() + text[1:]
        forms.update(text[:-1] + ending for ending in endings)
    return forms


def score(data, results):
    rules = {r['id']: r for r in json.loads((data / 'rules.json').read_text())['rules']}
    report = {'scope': 'Approved grammar forms preserve the complete selected meaning. Other forms need review.',
              'rules_sha256': hashlib.sha256((data / 'rules.json').read_bytes()).hexdigest(), 'splits': {}}
    for split in ('test', 'wording'):
        original = {r['id']: r for r in map(json.loads, (data / f'{split}.jsonl').read_text().splitlines())}
        generated = json.loads((results / f'{split}.json').read_text())['rows']
        by_kind, pairs, rows = {}, {}, []
        for result in generated:
            row = original[result['id']]
            if result['prefix'] != row['prefix']: raise ValueError('Evaluation data differs')
            approved = result['text'] in accepted_forms(row, rules[row['rule']]) and result['stopped']
            item = {**result, 'approved_meaning': approved}
            rows.append(item)
            by_kind.setdefault(row['kind'], []).append(approved)
            pairs.setdefault(row['pair'], []).append(approved)
        named = [r for r in rows if r['required_names']]
        report['splits'][split] = {'count': len(rows), 'approved_meaning': sum(r['approved_meaning'] for r in rows),
                                  'needs_review': sum(not r['approved_meaning'] for r in rows),
                                  'named_cases': len(named), 'all_names_in_named_cases': sum(r['all_names'] for r in named),
                                  'complete_pairs': sum(len(v) == 2 for v in pairs.values()),
                                  'approved_pairs': sum(len(v) == 2 and all(v) for v in pairs.values()),
                                  'kinds': {k: {'count': len(v), 'approved': sum(v)} for k, v in by_kind.items()},
                                  'rows': rows}
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    report = score(args.data, args.results)
    (args.results / 'meaning.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({s: {k: v for k, v in r.items() if k not in ('rows', 'kinds')}
                      for s, r in report['splits'].items()}, indent=2))
