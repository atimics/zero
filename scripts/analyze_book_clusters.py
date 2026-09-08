"""Attach exact corpus provenance to saved losses; bootstrap whole books."""
import argparse,bisect,hashlib,json,math
from pathlib import Path
import numpy as np

def book_ranges(ready,split):
    rows=[json.loads(line) for line in (ready/f'{split}.jsonl').read_text().splitlines()]
    reconstructed='\n\n'.join(r['text'] for r in rows)+'\n';raw=(ready/f'{split}.txt').read_bytes()
    if reconstructed.encode('ascii')!=raw:raise ValueError('Corpus ordering differs from scored source')
    ranges=[];offset=0
    for i,row in enumerate(rows):
        meta=row['metadata'];size=len(row['text'].encode('ascii'))+(2 if i<len(rows)-1 else 1)
        if ranges and ranges[-1]['book_id']==meta['book_id']:ranges[-1]['end']=offset+size
        else:ranges.append({'book_id':meta['book_id'],'author':meta['author'],'title':meta['title'],'start':offset,'end':offset+size})
        offset+=size
    return ranges,hashlib.sha256(raw).hexdigest()

def attach(short,long,ranges,offsets):
    if [(r['start_token'],r['bytes']) for r in short]!=[(r['start_token'],r['bytes']) for r in long]:raise ValueError('Loss rows are not paired')
    starts=[r['start'] for r in ranges];rows=[];excluded=[]
    for a,b in zip(short,long):
        t=a['start_token'];left=int(offsets[t]);right=int(offsets[t+64]);book=ranges[bisect.bisect_right(starts,left)-1]
        if right-left!=a['bytes']:raise ValueError('Byte denominator differs from token stream')
        r={'start_token':t,'bytes':a['bytes'],'short_nats':a['nats'],'long_nats':b['nats'],'context_crosses_book_boundary':int(offsets[t-961])<book['start'],'target_crosses_book_boundary':right>book['end'],**{k:book[k] for k in ['book_id','author','title']}}
        # Keep all scored input and target tokens within the assigned book.
        if int(offsets[t-961])<book['start'] or right>book['end']:excluded.append(r)
        else:rows.append(r)
    return rows,excluded

def aggregate(rows):
    n=sum(r['bytes'] for r in rows)
    a=sum(r['short_nats'] for r in rows)/math.log(2)/n;b=sum(r['long_nats'] for r in rows)/math.log(2)/n
    return {'short_bpb':a,'long_bpb':b,'long_minus_short_bpb':b-a,'windows':len(rows),'bytes':n,'books':len({r['book_id'] for r in rows}),'long_wins':sum(r['long_nats']<r['short_nats'] for r in rows)}

def clustered(rows,draws=20000,seed=1717):
    keys=sorted({r['book_id'] for r in rows});groups=[[r for r in rows if r['book_id']==k] for k in keys]
    delta=np.array([sum(r['long_nats']-r['short_nats'] for r in group)/math.log(2) for group in groups]);size=np.array([sum(r['bytes'] for r in group) for group in groups]);rng=np.random.default_rng(seed);samples=[]
    for start in range(0,draws,500):
        picks=rng.integers(len(keys),size=(min(500,draws-start),len(keys)));samples.extend((delta[picks].sum(1)/size[picks].sum(1)).tolist())
    return {'unit':'whole book, with replacement; all retained windows travel with their book','book_ids':keys,'draws':draws,'seed':seed,'percentile_95_interval':np.quantile(samples,[.025,.975]).tolist(),'conditional_scope':'Fixed trained weights and sampled books. Authors are not resampled; test contains one author. Eight test-book clusters give limited precision.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--tokens',type=Path,required=True);p.add_argument('--losses',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    short=json.loads((a.losses/'existing-window-rescore-5m-256-windows.json').read_text());long=json.loads((a.losses/'existing-window-rescore-5m-1024-windows.json').read_text());lengths=np.array(json.loads((a.tokens/'token_bytes.json').read_text()));result={'backend':short['backend'],'splits':{}}
    for split in ['validation','test']:
        ranges,sha=book_ranges(a.ready,split);ids=np.fromfile(a.tokens/f'{split}.bin',dtype='<u2');offsets=np.concatenate(([0],np.cumsum(lengths[ids],dtype=np.int64)))
        if int(offsets[-1])!=(a.ready/f'{split}.txt').stat().st_size:raise ValueError('Token/source length mismatch')
        rows,excluded=attach(short['splits'][split],long['splits'][split],ranges,offsets)
        result['splits'][split]={'source_sha256':sha,'pooled_retained':aggregate(rows),'all_windows_target_start_assignment':{'pooled':aggregate(rows+excluded),'book_bootstrap':clustered(rows+excluded),'by_author':{author:aggregate([r for r in rows+excluded if r['author']==author]) for author in sorted({r['author'] for r in rows+excluded})}},'by_author':{author:aggregate([r for r in rows if r['author']==author]) for author in sorted({r['author'] for r in rows})},'book_bootstrap':clustered(rows),'excluded_cross_book_windows':excluded,'windows':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({s:{k:v for k,v in d.items() if k not in ['windows','excluded_cross_book_windows']}|{'excluded_windows':len(d['excluded_cross_book_windows'])} for s,d in result['splits'].items()},indent=2))
if __name__=='__main__':main()
