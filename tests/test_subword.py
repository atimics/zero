import contextlib
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from subword_model import create,CONFIGS
from train_subword import evaluate

class SubwordTest(unittest.TestCase):
    def test_context_weights_and_causality(self):
        a=create(CONFIGS['5m-256']);b=create(CONFIGS['5m-1024'])
        self.assertEqual(sum(p.numel() for p in a.parameters()),5049600)
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        a.eval();x=torch.arange(12)[None,:];y=x.clone();y[0,8:]=100
        with torch.no_grad():torch.testing.assert_close(a(x)[:,:8],a(y)[:,:8])

    def test_attention_reference_and_gradients(self):
        c=dict(vocab=32,context=16,dim=16,heads=2,layers=2,ff=32)
        a=create(c);b=create(c);x=torch.arange(8)[None,:];target=(x+1)%32
        logits=a(x);F.cross_entropy(logits.flatten(0,1),target.flatten()).backward()
        def reference(q,k,v,is_causal):
            score=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
            mask=torch.ones(score.shape[-2:],dtype=torch.bool).triu(1)
            return score.masked_fill(mask,float('-inf')).softmax(-1)@v
        with patch('torch.nn.functional.scaled_dot_product_attention',reference):
            other=b(x);F.cross_entropy(other.flatten(0,1),target.flatten()).backward()
        torch.testing.assert_close(logits,other,atol=1e-6,rtol=1e-5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)

    def test_evaluation_targets_and_bytes(self):
        class Perfect(torch.nn.Module):
            def __init__(self,context):
                super().__init__();self.context=context;self.p=torch.nn.Parameter(torch.zeros(1));self.targets=[]
            def forward(self,x):
                self.targets.append(((x[0,-64:]+1)%128).tolist())
                return torch.nn.functional.one_hot((x+1)%128,128).float()*20
        data=torch.arange(10000)%128;lengths=torch.arange(128)%4+1
        a,b=Perfect(256),Perfect(1024)
        x=evaluate(a,data,lengths,4,contextlib.nullcontext);y=evaluate(b,data,lengths,4,contextlib.nullcontext)
        self.assertEqual(a.targets,b.targets);self.assertEqual(x,y);self.assertLess(x['bits_per_byte'],.0001)
        self.assertEqual(x['target_bytes'],sum(int(lengths[t]) for row in a.targets for t in row))

if __name__=='__main__':unittest.main()
