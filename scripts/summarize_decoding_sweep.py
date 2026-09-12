"""Summarize every setting and prepare a blind paired review of penalty 1.1."""
import argparse
import collections
import html
import json
import random
import re
from pathlib import Path


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def summarize(rows):
    by_prompt = {}
    for prompt in dict.fromkeys(r['prompt'] for r in rows):
        selected = [r for r in rows if r['prompt'] == prompt]
        bodies = [r['output'][len(prompt):] for r in selected]
        tokens = [words(b) for b in bodies]
        bigrams = [tuple(w[i:i+2]) for w in tokens for i in range(len(w)-1)]
        by_prompt[prompt] = {'samples': len(selected), 'mean_words': sum(map(len,tokens))/len(tokens),
                             'unique_outputs': len(set(bodies)), 'across_seed_distinct_bigram_ratio': len(set(bigrams))/max(1,len(bigrams)),
                             'mean_bigram_repeat_excess': sum(r['repetition']['2']['repeat_excess_rate'] for r in selected)/len(selected),
                             'mean_fourgram_repeat_excess': sum(r['repetition']['4']['repeat_excess_rate'] for r in selected)/len(selected)}
    return {'samples': len(rows), 'mean_bigram_repeat_excess': sum(r['repetition']['2']['repeat_excess_rate'] for r in rows)/len(rows),
            'mean_fourgram_repeat_excess': sum(r['repetition']['4']['repeat_excess_rate'] for r in rows)/len(rows),
            'by_prompt': by_prompt}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); a = p.parse_args()
    rows = [json.loads(line) for line in (a.directory/'samples.jsonl').read_text().splitlines()]
    if len(rows) != 576: raise ValueError('Expected the complete 576-output grid')
    keys = [(r['temperature'],r['top_k'],r['repetition_penalty'],r['prompt'],r['seed']) for r in rows]
    if len(set(keys)) != 576: raise ValueError('Duplicate grid cells')
    configs = sorted(set(k[:3] for k in keys))
    result = {'scope': 'One fixed checkpoint. Means weight each prompt/seed equally. Across-seed diversity pools eight outputs within a prompt. These measures are distinct from coherence ratings.', 'settings': []}
    for t,k,p in configs:
        r = summarize([r for r in rows if (r['temperature'],r['top_k'],r['repetition_penalty'])==(t,k,p)])
        result['settings'].append(dict(temperature=t,top_k=k,repetition_penalty=p,**r))
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # This pair was chosen from the previous French control, independently of
    # this grid's scores. The other settings remain exploratory.
    base = [r for r in rows if (r['temperature'],r['top_k'],r['repetition_penalty'])==(.7,40,1.0)]
    rng = random.Random(1701); rng.shuffle(base)
    key = []; markdown = ["# ZERO A/B review\n\nWhich continuation is better? Pick A, B, or skip. Review as many as you like. The case order and A/B assignments match the original packet.\n"]
    for i, r in enumerate(base,1):
        other = next(x for x in rows if (x['temperature'],x['top_k'],x['repetition_penalty'],x['prompt'],x['seed'])==(.7,40,1.1,r['prompt'],r['seed']))
        pair = [r,other]; rng.shuffle(pair)
        key.append({'case':i,'prompt':r['prompt'],'seed':r['seed'],'A_penalty':pair[0]['repetition_penalty'],'B_penalty':pair[1]['repetition_penalty']})
        markdown.append(f'## Case {i}\n\nOpening: {r["prompt"]}\n')
        for label, row in zip(['A','B'],pair):
            markdown.append(f'### {label}\n\n{row["output"]}\n')
        markdown.append('Pick: A / B / skip\n')
    (a.directory/'blind-review.md').write_text('\n'.join(markdown))
    (a.directory/'blind-review-key.json').write_text(json.dumps(key,indent=2)+'\n')
    from build_ab_review import write_review
    write_review(a.directory, rows, key)
    print(json.dumps([{k:v for k,v in row.items() if k!='by_prompt'} for row in result['settings']],indent=2))

if __name__ == '__main__': main()
