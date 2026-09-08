"""Generate a fixed eight-seed sample grid from both released 5M models."""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from subword_model import create,sample
from train_subword import PROMPTS
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2);tok=Tokenizer.from_file(str(ROOT/'experiments/subword-scaling/tokenizer.json'));rows=[]
for name in ['5m-256','5m-1024']:
    state=torch.load(ROOT/f'models/subword/{name}.pt',map_location='cpu',weights_only=True);model=create(state['config']);model.load_state_dict(state['model'])
    for seed in [1,2,3,4,5,6,7,8]:
        for prompt in PROMPTS:rows.append({'model':name,'seed':seed,'prompt':prompt,'output':sample(model,tok,prompt,128,seed)})
        print(name,seed,flush=True)
    (ROOT/'docs/subword/seed-grid.json').write_text(json.dumps({'backend':'CPU FP32','temperature':.7,'top_k':40,'generated_tokens':128,'samples':rows},indent=2)+'\n')
