import collections
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from boundary_data import roster, roster_digest, packs, forward
from boundary_common import identity
from diagnose_boundaries import mechanism_check, generate
from subword_model import create
from train_boundaries import train_core

CONFIG = dict(vocab=32,context=16,dim=16,heads=2,layers=2,ff=24)


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_exact_target_multiset_with_partial_pass_and_shuffle(self):
        records=[{'start':0,'tokens':7},{'start':7,'tokens':4},{'start':11,'tokens':6}]
        data=np.arange(17)
        expected=collections.Counter()
        remaining=37
        while remaining:
            for r in records:
                n=min(remaining,r['tokens']-1)
                expected.update(range(r['start']+1,r['start']+1+n));remaining-=n
                if not remaining:break
        digests=[]
        for shuffle in [False,True]:
            visits=roster(records,37,shuffle)
            digests.append(roster_digest(visits,canonical=True))
            rows=list(packs(data,records,visits,5)); actual=collections.Counter()
            for x,y,segments in rows:
                for a,b,s in zip(x,y,segments):
                    if b!=-100:
                        self.assertEqual(b,a+1); actual.update([int(b)])
            self.assertEqual(actual,expected)
            self.assertEqual(sum((y!=-100).sum() for _,y,_ in rows),37)
        self.assertEqual(digests[0],digests[1])
        self.assertEqual(roster(records,37,True),roster(records,37,True))
        self.assertNotEqual(roster(records,37,True),roster(records,37,False))

    def test_single_record_forward_and_gradients_match(self):
        a=create(CONFIG);b=create(CONFIG);x=torch.tensor([[1,2,3,4,5]])
        left=a(x);right=forward(b,x,torch.zeros_like(x),True)
        torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
        left.square().mean().backward();right.square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)

    def test_record_isolation_causality_cache_and_position_reset(self):
        model=create(CONFIG);x=torch.tensor([[1,2,3,4,5,6,7,8]])
        self.assertTrue(mechanism_check(model,x,'cpu')['passed'])
        segments=torch.tensor([[0,0,0,0,1,1,1,1]])
        combined=forward(model,x,segments,True)
        standalone=model(x[:,4:])
        torch.testing.assert_close(combined[:,4:],standalone,rtol=1e-5,atol=1e-6)
        # Changing the same record's past must still change its later predictions.
        changed=x.clone();changed[:,4]=9
        self.assertGreater(float((forward(model,changed,segments,True)[:,-1]-combined[:,-1]).abs().max().detach()),1e-5)

    def test_resume_restores_optimizer_dropout_and_best(self):
        records=[{'start':0,'tokens':9},{'start':9,'tokens':9}]; data=np.arange(18)
        visits=roster(records,70,True)
        config={'windows_per_update':1,'microbatch_windows':1,'selection_every':1,'dropout':.1}
        run={'isolate':True}
        def setup():
            torch.manual_seed(19);m=create(CONFIG);return m,torch.optim.AdamW(m.parameters(),lr=.0003)
        def score(m):
            m.eval();return {'bits_per_byte':float(sum(p.detach().square().sum() for p in m.parameters()))}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp); m,o=setup()
            done=train_core(m,o,data,records,visits,root/'full',config,run,score,'cpu')
            expected={k:v.clone() for k,v in m.state_dict().items()}
            m,o=setup();calls=0
            def interrupted(m):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('power loss')
                return score(m)
            with self.assertRaisesRegex(RuntimeError,'power loss'):
                train_core(m,o,data,records,visits,root/'resume',config,run,interrupted,'cpu')
            m,o=setup()
            resumed=train_core(m,o,data,records,visits,root/'resume',config,run,score,'cpu')
            self.assertEqual(done['target_presentations'],70)
            self.assertEqual(resumed['selected_step'],done['selected_step'])
            for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'identity'):
                train_core(m,o,data,records,visits,root/'resume',config,{'isolate':False},score,'cpu')

    def test_generation_resumes_and_rejects_corruption(self):
        class Tokenizer:
            def encode(self,text):return type('Encoding',(),{'ids':[1,2]})()
            def decode(self,ids):return 'hi' if ids==[1,2] else 'word '*len(ids)
        cases=[{'case_id':1,'prompt':'hi'},{'case_id':2,'prompt':'hi'}]
        settings={'greedy':{'temperature':1.,'top_k':1,'penalty':1.}}
        m=create({**CONFIG,'context':128})
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp)/'samples'
            with patch('diagnose_boundaries.sample',return_value=('unused',[1,2]+[3]*64)) as sampler:
                generate(m,Tokenizer(),cases,settings,output,{'bound':'same'},64,107,'cpu')
                self.assertEqual(sampler.call_count,2)
                generate(m,Tokenizer(),cases,settings,output,{'bound':'same'},64,107,'cpu')
                self.assertEqual(sampler.call_count,2)
            f=output/'case-greedy-001.json';row=json.loads(f.read_text());row['payload']['continuation']='changed'
            f.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'identity'):
                generate(m,Tokenizer(),cases,settings,output,{'bound':'same'},64,107,'cpu')

    def test_tiny_training_learns_repeated_sequence(self):
        torch.manual_seed(7);m=create(CONFIG);o=torch.optim.AdamW(m.parameters(),lr=.015)
        x=torch.tensor([[1,2,3,4]*4]);y=torch.tensor([[2,3,4,1]*4]);seg=torch.zeros_like(x)
        values=[]
        for _ in range(35):
            o.zero_grad();loss=torch.nn.functional.cross_entropy(forward(m,x,seg,True).flatten(0,1),y.flatten())
            values.append(float(loss.detach()));loss.backward();o.step()
        self.assertLess(values[-1],values[0]*.2)



class PackageTests(unittest.TestCase):
    def test_package_binds_worker_watchdog_and_every_payload(self):
        import tarfile
        from prepare_boundary_experiment import package
        from boundary_common import contract
        from canada_narrative import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); checkpoint=root/'checkpoints/BC-B';checkpoint.mkdir(parents=True)
            (checkpoint/'best.pt').write_bytes(b'checkpoint');(checkpoint/'result.json').write_text('{}')
            def prepare(delivery,output,region):
                output.mkdir();payload=root/'legacy';payload.write_bytes(b'legacy')
                with tarfile.open(output/'source.tar.gz','w:gz') as archive:archive.add(payload,arcname='legacy')
                (output/'user-data.template.sh').write_text('SOURCE_SHA='+digest(output/'source.tar.gz')+'\nshutdown -h +85\ntimeout 4500 "$PYTHON" scripts/aws_canada_pilot.py --delivery delivery --data prepared --output output\n')
                (output/'stack.json').write_text(json.dumps({'Resources':{'Watchdog':{'Properties':{'Code':{'ZipFile':'age >= 5100'}}}}}))
                return {'hourly_instance_usd':1.006,'source_files':{},'files':{n:'' for n in ['source.tar.gz','user-data.template.sh','stack.json']}}
            with patch('prepare_boundary_experiment.prepare_pilot',side_effect=prepare),patch('prepare_boundary_experiment.checkpoint_files',return_value={f'BC-B/{n}':checkpoint/n for n in ['best.pt','result.json']}):
                result=package(root,checkpoint.parent,root/'package')
            self.assertEqual(result['requested_budget_usd'],10)
            self.assertEqual(result['watchdog_maximum_age_seconds'],15000)
            script=(root/'package/user-data.template.sh').read_text()
            self.assertIn('shutdown -h +250',script);self.assertIn('timeout 14400',script)
            self.assertIn('aws_boundary_workload.py',script)
            self.assertIn('15000',(root/'package/stack.json').read_text())
            for n,h in result['files'].items():self.assertEqual(digest(root/'package'/n),h)
            with tarfile.open(root/'package/source.tar.gz') as archive:
                hashes=archive.extractfile('SHA256SUMS').read().decode().splitlines()
                self.assertEqual(len(hashes),len(archive.getmembers())-1)
                import hashlib
                for row in hashes:
                    h,n=row.split('  ');self.assertEqual(hashlib.sha256(archive.extractfile(n).read()).hexdigest(),h)


if __name__=='__main__':unittest.main()
