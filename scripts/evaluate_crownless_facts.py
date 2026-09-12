"""Compare free speech on reserved names, changed facts, and reserved wording."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from tune_crownless import read_split, save_json
from zero_torch import load


@torch.no_grad()
def generate_batch(model, examples, device, batch_size=16, max_chars=160):
    model.eval()
    result=[]
    for start in range(0,len(examples),batch_size):
        group=examples[start:start+batch_size]
        sequences=[list(e['prefix']) for e in group]
        outputs=[[] for _ in group]
        stopped=[False for _ in group]
        for _ in range(max_chars):
            active=[not stopped[i] and len(s)<=model.context for i,s in enumerate(sequences)]
            if not any(active): break
            length=max(min(len(s),model.context) for s in sequences)
            tokens=torch.zeros((len(group),length),dtype=torch.long)
            positions=[]
            for i,s in enumerate(sequences):
                n=min(len(s),model.context)
                tokens[i,:n]=torch.tensor(s[:n])
                positions.append(n-1)
            logits=model(tokens.to(device))[torch.arange(len(group),device=device),
                                           torch.tensor(positions,device=device)]
            logits[:,:10]=-torch.inf
            logits[:,11:32]=-torch.inf
            logits[:,127:]=-torch.inf
            next_tokens=logits.argmax(-1).cpu().tolist()
            for i,token in enumerate(next_tokens):
                if not active[i]: continue
                if token==10: stopped[i]=True
                else:
                    outputs[i].append(token)
                    sequences[i].append(token)
        for i,e in enumerate(group):
            result.append(dict(id=e['id'],kind=e['kind'],events=e['prefix'].decode(),
                               expected=e['target'].decode().rstrip('\n'),
                               generated=bytes(outputs[i]).decode(),stopped=stopped[i]))
    return result


def contains(text,name):
    return re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text,re.I) is not None


def measure(samples, audit):
    by_id={r['id']:r for r in audit}
    pairs=defaultdict(list)
    for sample in samples:
        row=by_id[sample['id']]
        required=row.get('required',{})
        sample['required']=required
        sample['required_names_match']=bool(required) and all(contains(sample['generated'],n) for n in required.values())
        sample['exact_reference']=sample['generated']==sample['expected']
        if 'pair_id' in row:
            sample['pair_id']=row['pair_id']
            sample['changed_field']=row['changed_field']
            sample['changed_value']=row['facts'][row['changed_field']]
            opposite=row['changed_to'] if row['facts'][row['changed_field']]==row['changed_from'] else row['changed_from']
            sample['changed_fact_match']=contains(sample['generated'],sample['changed_value']) and not contains(sample['generated'],opposite)
            pairs[row['pair_id']].append(sample)
    scored=[s for s in samples if s['required']]
    complete=[p for p in pairs.values() if len(p)==2]
    return dict(rows=len(samples),nonempty=sum(bool(s['generated']) for s in samples),
                stopped=sum(s['stopped'] for s in samples),exact_reference=sum(s['exact_reference'] for s in samples),
                required_names_match=sum(s['required_names_match'] for s in scored),required_names_scored=len(scored),
                changed_pairs_match=sum(all(s['changed_fact_match'] for s in pair) for pair in complete),pairs=len(complete),
                scope='Exact required-name presence and paired name changes. Event meaning and uncertainty require review.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',action='append',required=True,help='LABEL=CHECKPOINT')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    results={}
    for model_arg in args.model:
        label,path=model_arg.split('=',1)
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',label): p.error('Use a simple model label')
        model,_=load(path);model.to(args.device)
        scores={}
        for split in ['test','wording_test','editorial_test']:
            examples=read_split(args.data,split,model.context)
            audit=[json.loads(l) for l in (args.data/(split+'.audit.jsonl')).read_text().splitlines()]
            samples=generate_batch(model,examples,args.device)
            scores[split]=measure(samples,audit)
            save_json(args.output/(label+'-'+split+'.json'),samples)
            print(json.dumps(dict(model=label,split=split,**scores[split])),flush=True)
        results[label]=dict(checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),scores=scores)
        del model
    save_json(args.output/'comparison.json',results)


if __name__=='__main__': main()
