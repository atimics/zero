"""Paired window bootstrap of byte-weighted long minus short model loss."""
import argparse,json,math
from pathlib import Path
import numpy as np

def bootstrap(a,b,draws=20000,seed=1701):
    if [r['start_token'] for r in a]!=[r['start_token'] for r in b] or [r['bytes'] for r in a]!=[r['bytes'] for r in b]:raise ValueError('Targets are not paired')
    lengths=np.array([r['bytes'] for r in a]);delta=np.array([y['nats']-x['nats'] for x,y in zip(a,b)])/math.log(2)
    generator=np.random.default_rng(seed);estimates=[]
    for offset in range(0,draws,500):
        sample=generator.integers(len(a),size=(min(500,draws-offset),len(a)))
        estimates.extend((delta[sample].sum(1)/lengths[sample].sum(1)).tolist())
    return {'long_minus_short_bpb':float(delta.sum()/lengths.sum()),'paired_window_95_percentile_interval':np.quantile(estimates,[.025,.975]).tolist(),'long_wins':int((delta<0).sum()),'short_wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'windows':len(a),'bootstrap_draws':draws,'bootstrap_seed':seed,'caveat':'Conditional on these trained weights and sampled windows. Windows from the same book can be correlated; this interval excludes training-seed and author-sampling uncertainty.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();short=json.loads((a.directory/'5m-256-windows.json').read_text());long=json.loads((a.directory/'5m-1024-windows.json').read_text());result={'backend':short['backend'],'splits':{s:bootstrap(short['splits'][s],long['splits'][s]) for s in ['validation','test']}};(a.directory/'paired-bootstrap.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
