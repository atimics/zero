"""Expand name shapes and replay original game accounts during Crownless tuning."""
import argparse
import json
import random
import re
from pathlib import Path

from build_crownless_facts import PATTERNS, build, pools, sha


def game_names(source):
    result={k:set() for k in ['place','person','faction','dragon','object']}
    for line in (source/'train.audit.jsonl').read_text().splitlines():
        row=json.loads(line)
        for kind,pattern in PATTERNS:
            m=re.fullmatch(pattern,row['input']['account']) if kind==row['input']['kind'] else None
            if m:
                for key,value in m.groupdict().items():
                    role='faction' if key=='other' else key
                    if role in result: result[role].add(value)
                break
    return result


def wide_pools(source,seed):
    previous=pools(); known=game_names(source)
    rng=random.Random(seed)
    syllables='al an ar ash ba bel ber bra bri ca car cor da dar del dor el em en er fa fen fir for gal gar gla glo hal har hol il in ir jar ka kel kir la len li lor ma mar mel mer mi mor na nel nor ol or ra ran ren ri ril ro ros sa sar sel sha sil ta tal tel tha thor ti tor ul va val ven ver vi vor wa wen wil ya yen yor za zel'.split()
    def word():
        return ''.join(rng.choices(syllables,k=rng.choice([2,2,3]))).capitalize()
    roles=['place','person','faction','dragon','object']
    values={r:set().union(*(set(previous[s][r]) for s in previous),known[r]) for r in roles}
    for _ in range(16000):
        place=word()+rng.choice(['ford','gate','barrow','spire','wick','mere','watch','bank',''])
        values['place'].add(place)
        values['person'].add(word()+' '+rng.choice([word(),word()+'ward',word()+'mender',word()+'wright']))
        values['faction'].add(rng.choice(['','The '])+word()+' '+rng.choice(['Parliament','Tithe','Company','Court','Republic','Throne','Knives','March','Oath']))
        values['dragon'].add(word()+rng.choice(['','wing',' the Unappeased',' the Pale',' the Red']))
        values['object'].add(rng.choice(['',place+' '])+word()+' '+rng.choice(['Cup','Icon','Crown','Reliquary','Torc','Bell']))
    result={s:{} for s in previous}
    for role,names in values.items():
        for s in result: result[s][role]=[]
        for n in sorted(names):
            bucket=int(sha(n.encode())[:8],16)%10
            split='train' if n in known[role] or bucket<8 else 'validation' if bucket==8 else 'test'
            result[split][role].append(n)
    for s in result:
        for key in ['subject','direction']: result[s][key]=previous[s][key]
        result[s]['other']=result[s]['faction']
    return result,known


def mixed(source,output,pairs=12000,seed=73):
    names,known=wide_pools(source,seed)
    manifest=build(source,output,pairs,seed,name_pools=names)
    text=bytearray((output/'train.txt').read_bytes())
    audit=[json.loads(l) for l in (output/'train.audit.jsonl').read_text().splitlines()]
    original=(source/'train.txt').read_bytes()
    original_rows=[json.loads(l) for l in (source/'train.audit.jsonl').read_text().splitlines()]
    for repeat in range(2):
        base=len(text); text.extend(original)
        for row in original_rows:
            r=dict(row)
            r['id']=f"replay-{repeat}-"+row['id']
            r['text_start']+=base; r['output_start']+=base
            r['replay']=True
            audit.append(r)
    (output/'train.txt').write_bytes(text)
    (output/'train.audit.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in audit))
    manifest.update(version=4,replay_rows=2*len(original_rows),name_seed=seed,
                    known_training_names={k:len(v) for k,v in known.items()},
                    name_pools={s:{k:len(v) for k,v in ps.items()} for s,ps in names.items()},
                    generator_sha256={n:sha((Path(__file__).parent/n).read_bytes()) for n in ['build_crownless_facts.py','build_crownless_mixed.py']})
    manifest['splits']['train']=dict(rows=len(audit),bytes=len(text),sha256=sha(text))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=12000)
    p.add_argument('--seed',type=int,default=73)
    a=p.parse_args()
    if a.pairs<12:p.error('Use at least twelve pairs')
    print(json.dumps(mixed(a.source,a.output,a.pairs,a.seed),indent=2))
