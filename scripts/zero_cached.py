"""Reuse ZERO attention state while completing a batch of event lists."""
import torch
from torch.nn import functional as F


class CachedZero:
    def __init__(self, model):
        self.model=model
        self.cache=[]

    def prefill(self,tokens,lengths):
        m=self.model
        self.lengths=lengths.clone()
        self.valid=torch.arange(tokens.shape[1],device=tokens.device)[None,:]<lengths[:,None]
        self.cache=[]
        value=F.embedding(tokens,m.weights[0])
        for offset in range(1,len(m.weights)-1,8):
            n1,wq,wk,wv,wo,n2,w1,w2=m.weights[offset:offset+8]
            normal=m.norm(value,n1)
            q,k=m.rotate(F.linear(normal,wq)),m.rotate(F.linear(normal,wk))
            v=F.linear(normal,wv).reshape(*tokens.shape,m.heads,-1).transpose(1,2)
            self.cache.append((k,v))
            attention=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            attention=attention.transpose(1,2).reshape(*tokens.shape,m.dim)
            value=value+F.linear(attention,wo)
            value=value+F.linear(F.gelu(F.linear(m.norm(value,n2),w1),approximate='tanh'),w2)
        last=value[torch.arange(len(lengths),device=tokens.device),lengths-1]
        return F.linear(m.norm(last,m.weights[-1]),m.weights[0])

    def step(self,tokens):
        m=self.model
        value=F.embedding(tokens[:,None],m.weights[0])
        positions=self.lengths.clamp(max=m.context-1)
        c=m.rope_cos[positions][:,None,None,:]
        s=m.rope_sin[positions][:,None,None,:]
        def rotate(x):
            x=x.reshape(len(tokens),1,m.heads,-1).transpose(1,2)
            even,odd=x[...,0::2],x[...,1::2]
            return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)
        self.valid=torch.cat((self.valid,torch.ones((len(tokens),1),dtype=torch.bool,device=tokens.device)),dim=1)
        for layer,offset in enumerate(range(1,len(m.weights)-1,8)):
            n1,wq,wk,wv,wo,n2,w1,w2=m.weights[offset:offset+8]
            normal=m.norm(value,n1)
            q,k=rotate(F.linear(normal,wq)),rotate(F.linear(normal,wk))
            v=F.linear(normal,wv).reshape(len(tokens),1,m.heads,-1).transpose(1,2)
            old_k,old_v=self.cache[layer]
            k,v=torch.cat((old_k,k),dim=2),torch.cat((old_v,v),dim=2)
            self.cache[layer]=(k,v)
            attention=F.scaled_dot_product_attention(q,k,v,attn_mask=self.valid[:,None,None,:])
            attention=attention.transpose(1,2).reshape(len(tokens),1,m.dim)
            value=value+F.linear(attention,wo)
            value=value+F.linear(F.gelu(F.linear(m.norm(value,n2),w1),approximate='tanh'),w2)
        self.lengths+=1
        return F.linear(m.norm(value[:,0],m.weights[-1]),m.weights[0])
