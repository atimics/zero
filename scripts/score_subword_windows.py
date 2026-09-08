"""Save each held-out window's loss before paired statistical analysis."""
import argparse,contextlib,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from subword_model import create

def score(model,data,lengths,windows,amp):
    rows=[]
    for start in np.linspace(1024,len(data)-64,windows,dtype=np.int64):
        prefix=model.context-64
        x=data[int(start)-prefix-1:int(start)+63].long()[None,:]
        target=data[int(start):int(start)+64].long()
        with torch.no_grad(),amp():loss=F.cross_entropy(model(x)[0,-64:].float(),target,reduction='sum').item()
        rows.append({'start_token':int(start),'nats':loss,'bytes':int(lengths[target].sum())})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');a=p.parse_args()
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    amp=(lambda:torch.autocast(a.device,dtype=torch.bfloat16)) if a.precision=='bf16' else contextlib.nullcontext
    lengths=torch.tensor(json.loads((a.data/'token_bytes.json').read_text()),device=a.device)
    for name in ['5m-256','5m-1024']:
        path=a.models/(name+'.pt');state=torch.load(path,map_location='cpu',weights_only=True);model=create(state['config']).to(a.device);model.load_state_dict(state['model']);model.eval()
        result={'model':name,'backend':a.device+' '+a.precision,'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'splits':{}}
        for split in ['validation','test']:
            data=torch.tensor(np.fromfile(a.data/(split+'.bin'),dtype='<u2').astype('int32'),device=a.device)
            result['splits'][split]=score(model,data,lengths,1024,amp);print(name,split,flush=True)
        (a.output/(name+'-windows.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
