"""Report registered paired training-seed statistics and proxy diagnostics."""
import argparse,json,math
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from bootstrap_subword import bootstrap
from subword_diagnostics import analyze
SEEDS=[7,11,19,31,43]

def paired(values):
    d=np.asarray(values,dtype=float)
    if len(d)!=5:return {'complete':False,'paired_deltas':values,'required_pairs':5}
    mean=float(d.mean());sd=float(d.std(ddof=1));radius=2.7764451051977987*sd/math.sqrt(5);interval=[mean-radius,mean+radius]
    return {'complete':True,'paired_deltas':values,'mean_long_minus_comparator_bpb':mean,'sample_standard_deviation':sd,'paired_t_95_interval':interval,'degrees_of_freedom':4,'direction':'long_lower' if interval[1]<0 else 'long_higher' if interval[0]>0 else 'unresolved','practical_equivalence_at_0_01':interval[0]>-.01 and interval[1]<.01,'assumption':'Paired training-seed differences are treated as approximately normal; five pairs give limited precision.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--prefix-data',type=Path,required=True);p.add_argument('--diagnostics',type=Path,required=True);a=p.parse_args();result={'seeds':SEEDS,'comparisons':{}}
    for comparator in ['5m-256','5m-wide-256']:
        deltas=[];author_deltas=[];by_seed=[]
        for seed in SEEDS:
            root=a.output/f'seed-{seed}';paths=[root/n/'result.json' for n in [comparator,'5m-1024']]
            if not all(p.exists() for p in paths):continue
            records=[json.loads(p.read_text()) for p in paths];delta=records[1]['test']['bits_per_byte']-records[0]['test']['bits_per_byte'];deltas.append(delta);entry={'seed':seed,'test_delta':delta}
            author_paths=[root/n/'extra-authors.json' for n in [comparator,'5m-1024']]
            if all(p.exists() for p in author_paths):
                authors=[json.loads(p.read_text()) for p in author_paths];ds=[delta]+[authors[1][n]['bits_per_byte']-authors[0][n]['bits_per_byte'] for n in ['shelley','stoker']];author_deltas.append(float(np.mean(ds)));entry['author_deltas']=dict(zip(['alcott','shelley','stoker'],ds))
            wp=[root/n/'window-losses.json' for n in [comparator,'5m-1024']]
            if all(p.exists() for p in wp):
                windows=[json.loads(p.read_text()) for p in wp];entry['conditional_window_bootstrap']=bootstrap(windows[0]['test'],windows[1]['test'])
            by_seed.append(entry)
        result['comparisons'][comparator]={'primary_alcott':paired(deltas),'secondary_three_author_mean':paired(author_deltas),'by_seed':by_seed}
    names=json.loads((a.diagnostics/'source-names.json').read_text());diagnostics={}
    for grid in sorted(a.output.glob('**/seed-grid.json')):
        prefix='prefix-space' in str(grid);root=a.prefix_data if prefix else a.data;index=json.loads((a.diagnostics/('prefix-token-placement.json' if prefix else 'token-placement.json')).read_text());tok=Tokenizer.from_file(str(root/'tokenizer.json'))
        diagnostics[str(grid.relative_to(a.output))]=analyze(json.loads(grid.read_text())['samples'],index,names,tok)
    (a.output/'generation-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n');(a.output/'replication-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparisons'],indent=2))
if __name__=='__main__':main()
