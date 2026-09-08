"""Score the character baseline on exactly the subword experiment's targets."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from zero_torch import load

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);model,_=load(a.checkpoint);model=model.to(a.device).eval();tok=Tokenizer.from_file(str(a.data/'tokenizer.json'));result={}
    for split in ['validation','test']:
        ids=np.fromfile(a.data/f'{split}.bin',dtype='<u2');total=0.;count=0
        for start in np.linspace(1024,len(ids)-64,1024,dtype=np.int64):
            target=tok.decode(ids[start:start+64].tolist()).encode('ascii')
            prefix=tok.decode(ids[start-1024:start].tolist()).encode('ascii')
            if len(target)>model.context:raise ValueError('Target span exceeds character context')
            raw=(prefix+target)[-model.context-1:];x=torch.tensor(list(raw),device=a.device)
            logits=model(x[:-1][None,:])[0,-len(target):]
            total+=F.cross_entropy(logits, x[-len(target):],reduction='sum').item();count+=len(target)
        result[split]={'bits_per_byte':total/math.log(2)/count,'target_bytes':count,'target_tokens':65536,'windows':1024}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
