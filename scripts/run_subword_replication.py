"""Execute the registered seed pairs, width arm and prefix-space experiment."""
import argparse,contextlib,hashlib,json,math,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tokenizers import Tokenizer
import train_subword as train
from subword_model import create,sample
from score_subword_windows import score
from prefix_comparison import compare

SEEDS=[7,11,19,31,43]
def load_data(path):
    manifest=json.loads((path/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:raise ValueError('Data hash mismatch')
    data={s:torch.tensor(np.fromfile(path/f'{s}.bin',dtype='<u2').astype('int32'),device='cuda') for s in ['train','validation','test']}
    lengths=torch.tensor(json.loads((path/'token_bytes.json').read_text()),device='cuda')
    return data,lengths,Tokenizer.from_file(str(path/'tokenizer.json'))

@torch.no_grad()
def extras(model,path,lengths):
    result={}
    for author in ['shelley','stoker']:
        ids=torch.tensor(np.fromfile(path/f'{author}.bin',dtype='<u2').astype('int32'),device='cuda');starts=json.loads((path/f'{author}-starts.json').read_text());rows=[]
        for start in starts:
            prefix=model.context-64;x=ids[start-prefix-1:start+63].long()[None,:];target=ids[start:start+64].long()
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(x)[0,-64:].float(),target,reduction='sum').item()
            rows.append({'start_token':start,'nats':loss,'bytes':int(lengths[target].sum())})
        result[author]={'bits_per_byte':sum(r['nats'] for r in rows)/math.log(2)/sum(r['bytes'] for r in rows),'windows':rows}
    return result

def enrich(directory,data,lengths,tok,extra):
    state=torch.load(directory/'best.pt',map_location='cuda',weights_only=False);model=create(state['config']).cuda().eval();model.load_state_dict(state['model'])
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    windows={s:score(model,data[s],lengths,1024,amp) for s in ['validation','test']}
    train.save_json(directory/'window-losses.json',windows);train.save_json(directory/'extra-authors.json',extras(model,extra,lengths))
    rows=[]
    for seed in range(1,9):
        for prompt in train.PROMPTS:
            with amp():text,ids=sample(model,tok,prompt,128,seed,return_tokens=True)
            rows.append({'seed':seed,'prompt':prompt,'output':text,'token_ids':ids,'prompt_tokens':len(tok.encode(prompt).ids)})
    train.save_json(directory/'seed-grid.json',{'backend':'CUDA BF16','samples':rows})
    del model,state;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--prefix-data',type=Path,required=True);p.add_argument('--extra',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args();torch.set_num_threads(2)
    data,lengths,tok=load_data(a.data)
    args=SimpleNamespace(device='cuda',tokens_per_step=8192,small_tokens=100000000,large_tokens=800000000,report=100,selection_windows=64,final_windows=1024,deadline=a.deadline)
    for seed in SEEDS:
        args.seed=seed;args.output=a.output/f'seed-{seed}'
        for name in ['5m-256','5m-1024']:
            train.run(name,args,data,lengths,tok);enrich(args.output/name,data,lengths,tok,a.extra)
    for seed in SEEDS:
        args.seed=seed;args.output=a.output/f'seed-{seed}'
        train.run('5m-wide-256',args,data,lengths,tok);enrich(args.output/'5m-wide-256',data,lengths,tok,a.extra)
    data,lengths,tok=load_data(a.prefix_data);args.seed=7;args.output=a.output/'prefix-space'
    train.run('5m-256',args,data,lengths,tok)
    # The variant's native aggregate uses its own tokens; the paired raw-byte
    # comparison below provides the cross-tokenizer comparison.
    state=torch.load(args.output/'5m-256/best.pt',map_location='cuda',weights_only=False);model=create(state['config']).cuda().eval();model.load_state_dict(state['model']);rows=[]
    for seed in range(1,9):
        for prompt in train.PROMPTS:
            with torch.autocast('cuda',dtype=torch.bfloat16):text,ids=sample(model,tok,prompt,128,seed,return_tokens=True)
            rows.append({'seed':seed,'prompt':prompt,'output':text,'token_ids':ids,'prompt_tokens':len(tok.encode(prompt).ids)})
    train.save_json(args.output/'5m-256/seed-grid.json',{'backend':'CUDA BF16','samples':rows})
    del model,state;torch.cuda.empty_cache()
    compare([a.output/'seed-7/5m-256/best.pt',a.output/'prefix-space/5m-256/best.pt'],a.data,a.prefix_data,a.extra,a.output/'prefix-comparison.json')
    train.save_json(a.output/'complete.json',{'training_seeds':SEEDS,'completed_at':time.time(),'runs':16})
if __name__=='__main__':main()
