"""Describe source-name mixing and exact word repetition in saved samples."""
import argparse
import collections
import json
import re
from pathlib import Path


def repetition(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    result = {}
    for n in [1, 2, 3, 4]:
        counts = collections.Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        total = sum(counts.values())
        phrase, count = counts.most_common(1)[0] if counts else ((), 0)
        result[str(n)] = {'positions': total, 'max_count': count, 'max_ngram': ' '.join(phrase),
                          'max_frequency': count / max(1, total),
                          'repeat_excess_rate': sum(v-1 for v in counts.values()) / max(1, total)}
    return result


def measure(samples, names):
    rows = []
    for row in samples:
        body = row['output'][len(row['prompt']):]
        words = set(re.findall(r'\b[A-Z][a-z]{2,}\b', body))
        matches = {w: list(names['names'][w]) for w in sorted(words) if w in names['names']}
        books = sorted({b for ids in matches.values() for b in ids})
        rows.append({'prompt': row['prompt'], 'matched_names': matches, 'source_books': books,
                     'distinct_source_books': len(books), 'capitalized_words': sorted(words),
                     'coverage': len(matches) / max(1, len(words)), 'repetition': repetition(body)})
    return {'mean_distinct_source_books': sum(r['distinct_source_books'] for r in rows)/len(rows),
            'passages_with_multiple_source_books': sum(r['distinct_source_books']>1 for r in rows), 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--names', type=Path, required=True)
    p.add_argument('--historical', type=Path, required=True)
    a = p.parse_args()
    names = json.loads(a.names.read_text())
    result = {'method': 'Generated continuation only; case-folded word n-grams, punctuation ignored. Max frequency = largest count / all n-gram positions. Repeat excess = repeated occurrences beyond first / all positions. Exact lexical measures miss paraphrases and do not distinguish useful repetition.', 'models': {}}
    for name in ['5m-256', '5m-1024', '50m']:
        samples = json.loads((a.directory / ('samples-'+name+'.json')).read_text())['samples']
        result['models'][name] = measure(samples, names)
    control = json.loads((a.directory / 'repetition-control.json').read_text())['samples']
    result['penalty_control'] = measure(control, names)
    historical = json.loads(a.historical.read_text())
    result['historical_gpu_5m'] = {name: measure(historical['models'][name]['samples'], names) for name in ['5m-256', '5m-1024']}
    result['sara_in_frozen_index'] = names['names'].get('Sara')
    (a.directory / 'sample-diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: {key: val for key, val in v.items() if key != 'rows'} for k,v in result['models'].items()},indent=2))
    print('Sara:', result['sara_in_frozen_index'])

if __name__ == '__main__':
    main()
