"""ZERO architecture with configurable vocabulary, width, and context."""
import math
import numpy as np
import torch
from zero_torch import Zero

CONFIGS = {
    '5m-256': dict(vocab=2048,context=256,dim=256,heads=8,layers=6,ff=960),
    '5m-1024': dict(vocab=2048,context=1024,dim=256,heads=8,layers=6,ff=960),
    '5m-wide-256': dict(vocab=2048,context=256,dim=384,heads=12,layers=3,ff=1080),
    '50m-1024': dict(vocab=2048,context=1024,dim=640,heads=10,layers=10,ff=2560),
}

def create(config, seed=7):
    c=config; rng=np.random.default_rng(seed); d=c['dim']; arrays=[]
    arrays.append(rng.normal(0,.02,(c['vocab'],d)).astype('float32'))
    for _ in range(c['layers']):
        arrays.append(np.ones(d,dtype='float32'))
        for i in range(4):
            arrays.append(rng.normal(0,.02 / (math.sqrt(2*c['layers']) if i==3 else 1),(d,d)).astype('float32'))
        arrays.append(np.ones(d,dtype='float32'))
        arrays.append(rng.normal(0,.02,(c['ff'],d)).astype('float32'))
        arrays.append(rng.normal(0,.02/math.sqrt(2*c['layers']),(d,c['ff'])).astype('float32'))
    arrays.append(np.ones(d,dtype='float32'))
    header=[b'ZEROLM2\0',3,c['vocab'],c['context'],d,c['heads'],c['layers'],c['ff'],len(arrays),1,0,seed]
    return Zero(header,arrays)

@torch.no_grad()
def sample(model, tokenizer, prompt, count=128, seed=7, return_tokens=False, repetition_penalty=1.0):
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1")
    model.eval(); device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(seed)
    ids=tokenizer.encode(prompt).ids
    for _ in range(count):
        x=torch.tensor([ids[-model.context:]],device=device)
        scores=model(x)[0,-1].float()
        # Match the C sampler: divide recent-token probability by the penalty
        # before temperature and top-k; each of the last 64 tokens counts once.
        if repetition_penalty != 1.0:
            scores[list(set(ids[-64:]))] -= math.log(repetition_penalty)
        scores=scores/.7
        values,indices=scores.topk(40)
        token=indices[torch.multinomial(values.softmax(-1),1,generator=generator)].item();ids.append(token)
    text=tokenizer.decode(ids)
    return (text,ids) if return_tokens else text
