"""Sample an exported training checkpoint with its frozen tokenizer."""
import argparse
import json
import torch
from tokenizers import Tokenizer
from subword_model import create,sample
from train_subword import PROMPTS
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('tokenizer');p.add_argument('--prompt');p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2)
state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=create(state['config']).to(a.device);model.load_state_dict(state['model']);tokenizer=Tokenizer.from_file(a.tokenizer)
print(json.dumps({'step':state['step'],'samples':[{'prompt':p,'output':sample(model,tokenizer,p)} for p in ([a.prompt] if a.prompt else PROMPTS)]},indent=2))
