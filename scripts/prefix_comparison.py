"""Compare tokenizer variants on shared raw target spans and token boundaries."""
import json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from subword_model import create

@torch.no_grad()
def compare(paths,data,prefix_data,extra,output):
    tokenizers=[Tokenizer.from_file(str(p/'tokenizer.json')) for p in [data,prefix_data]]
    models=[]
    for path in paths:
        state=torch.load(path,map_location='cuda',weights_only=False);model=create(state['config']).cuda().eval();model.load_state_dict(state['model']);models.append(model)
    results={}
    for split,root in [('test',data),('shelley',extra),('stoker',extra)]:
        ids=np.fromfile(root/f'{split}.bin',dtype='<u2');starts=json.loads((root/f'{split}-starts.json').read_text()) if root==extra else np.linspace(1024,len(ids)-64,1024,dtype=np.int64)
        rows=[]
        for start in starts:
            prefix=tokenizers[0].decode(ids[int(start)-1024:int(start)].tolist());target=tokenizers[0].decode(ids[int(start):int(start)+64].tolist());text=prefix+target
            enc=[t.encode(text) for t in tokenizers]
            boundaries=[{v for a,b in e.offsets for v in [a,b]} for e in enc];common=boundaries[0]&boundaries[1]
            left=min(v for v in common if v>=len(prefix));right=max(v for v in common if v<=len(text))
            if right<=left:raise ValueError('No shared nonempty target')
            losses=[]
            for model,e in zip(models,enc):
                target_indices=[i for i,(a,b) in enumerate(e.offsets) if a>=left and b<=right and b>a]
                first,last=target_indices[0],target_indices[-1]+1
                if target_indices!=list(range(first,last)):raise ValueError('Noncontiguous targets')
                count=last-first
                if count>=model.context:raise ValueError('Target exceeds context')
                begin=max(0,last-1-model.context);x=torch.tensor(e.ids[begin:last],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(x[:-1][None,:])[0,-count:].float()
                losses.append(F.cross_entropy(scores,x[-count:],reduction='sum').item())
            rows.append({'start_token':int(start),'bytes':right-left,'trimmed_bytes':len(target)-(right-left),'nats':losses})
        total=sum(r['bytes'] for r in rows);results[split]={'bits_per_byte':[sum(r['nats'][i] for r in rows)/math.log(2)/total for i in range(2)],'target_bytes':total,'windows':rows}
    output.write_text(json.dumps({'method':'Both models score the same decoded raw target bytes, trimmed to boundaries shared by both tokenizers. Loss is summed over full target tokens; synthetic prefix spaces are outside targets.','splits':results},indent=2)+'\n')
