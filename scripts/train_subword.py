"""Run matched-text context experiments followed by a larger ZERO model."""
import argparse
import contextlib
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from subword_model import CONFIGS, create, sample

PROMPTS=['Once upon a time,','The little girl opened the door and','The captain looked across the sea.','The warren is ','Fund uncertainty. ','Who is Mara?']

def save_json(path, value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def checkpoint(path, model, optimizer, step, config, generator):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                    rng=generator.get_state(),cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),temporary)
    temporary.replace(path)

@torch.no_grad()
def evaluate(model,data,lengths,sequences,amp):
    # Identical target tokens for both context lengths; prefix alone varies.
    model.eval(); total=0.; byte_count=0; target_count=64
    starts=np.linspace(1024,len(data)-target_count,sequences,dtype=np.int64)
    device=next(model.parameters()).device
    for start in starts:
        prefix=min(model.context-target_count,1024)
        window=data[int(start)-prefix-1:int(start)+target_count-1].long().unsqueeze(0)
        targets=data[int(start):int(start)+target_count].long()
        with amp():
            scores=model(window)[0,-target_count:]
            loss=F.cross_entropy(scores.float(),targets,reduction='sum')
        total+=loss.item();byte_count+=int(lengths[targets].sum())
    model.train()
    return {'bits_per_byte':total/math.log(2)/byte_count,'target_bytes':byte_count,
            'target_tokens':sequences*target_count,'windows':sequences}


def run(name,args,data,lengths,tokenizer):
    config=CONFIGS[name];device=args.device
    output=args.output/name;output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(7); model=create(config).to(device)
    groups=[{'params':[w for w in model.weights if w.ndim==2],'weight_decay':.01},
            {'params':[w for w in model.weights if w.ndim==1],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=.0003,betas=(.9,.999),eps=1e-8,fused=device=='cuda')
    amp=(lambda:torch.autocast('cuda',dtype=torch.bfloat16)) if device=='cuda' else contextlib.nullcontext
    generator=torch.Generator(device=device).manual_seed(71)
    block=args.tokens_per_step
    if block%config['context']: raise ValueError('Batch must divide evenly into contexts')
    target=args.large_tokens if name.startswith('50m') else args.small_tokens
    steps=math.ceil(target/block);warmup=min(2000,max(1,steps//50));best=float('inf');history=[]
    started=time.monotonic();train_sum=0.;interval=0
    for step in range(1,steps+1):
        if time.time()>args.deadline-180: raise TimeoutError('Deadline reached; earlier saved checkpoints retained')
        # Every run draws the same contiguous token block per update. Context
        # changes only its division into sequences, preserving target exposure.
        start=torch.randint(len(data['train'])-block,(),generator=generator,device=device)
        ids=data['train'][start+torch.arange(block+1,device=device)].long()
        x=ids[:-1].reshape(-1,config['context']);y=ids[1:].reshape_as(x)
        optimizer.zero_grad(set_to_none=True)
        with amp(): loss=F.cross_entropy(model(x,dropout=.1).flatten(0,1),y.flatten())
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr=.0003*min(1,step/warmup)
        if step>warmup:lr*=.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.step();train_sum+=loss.detach().item();interval+=1
        if step%args.report==0 or step==steps:
            elapsed=time.monotonic()-started
            validation=evaluate(model,data['validation'],lengths,args.selection_windows,amp)
            row={'step':step,'steps':steps,'tokens':step*block,'train_loss':train_sum/interval,
                 'validation':validation,'elapsed_seconds':elapsed,'tokens_per_second':step*block/elapsed}
            history.append(row);print(name,json.dumps(row),flush=True);save_json(output/'history.json',history)
            checkpoint(output/'last.pt',model,optimizer,step,config,generator)
            if validation['bits_per_byte']<best:
                best=validation['bits_per_byte'];checkpoint(output/'best.pt',model,optimizer,step,config,generator)
            train_sum=0.;interval=0
            if step==args.report:
                estimate=(time.monotonic()-started)*(steps-step)/step
                if estimate>(args.deadline-time.time()-180)*.85:
                    raise TimeoutError(f'{name} needs about {estimate/3600:.2f} more hours; saved calibration checkpoint')
    state=torch.load(output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model'])
    result={'name':name,'config':config,'parameters':sum(p.numel() for p in model.parameters()),
            'steps':steps,'token_presentations':steps*block,'selected_step':state['step'],
            'elapsed_seconds':time.monotonic()-started,
            'validation':evaluate(model,data['validation'],lengths,args.final_windows,amp),
            'test':evaluate(model,data['test'],lengths,args.final_windows,amp)}
    with amp(): samples=[{'prompt':p,'output':sample(model,tokenizer,p)} for p in PROMPTS]
    save_json(output/'samples.json',samples);save_json(output/'result.json',result)
    print(json.dumps(result),flush=True)
    del optimizer,model,state
    if device=='cuda':torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runs',nargs='+',default=list(CONFIGS));p.add_argument('--device',default='cuda');p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--small-tokens',type=int,default=100_000_000);p.add_argument('--large-tokens',type=int,default=800_000_000)
    p.add_argument('--tokens-per-step',type=int,default=8192);p.add_argument('--report',type=int,default=100)
    p.add_argument('--selection-windows',type=int,default=64);p.add_argument('--final-windows',type=int,default=1024);args=p.parse_args()
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 GPU required')
    manifest=json.loads((args.data/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((args.data/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Hash mismatch: {name}')
    data={s:torch.tensor(np.fromfile(args.data/f'{s}.bin',dtype='<u2').astype('int32'),device=args.device) for s in ['train','validation','test']}
    lengths=torch.tensor(json.loads((args.data/'token_bytes.json').read_text()),device=args.device)
    tokenizer=Tokenizer.from_file(str(args.data/'tokenizer.json'))
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.runs:run(name,args,data,lengths,tokenizer)

if __name__=='__main__':main()
