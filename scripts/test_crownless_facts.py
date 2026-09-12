import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from build_crownless_facts import pools, render, build, WORDING
from evaluate_crownless_facts import contains, measure, generate_batch
from tune_crownless import generate, read_split
from zero_torch import Zero


class FactTests(unittest.TestCase):
    def test_names_are_disjoint(self):
        ps=pools()
        for role in ['place','person','dragon','faction','object']:
            a,b,c=[set(ps[s][role]) for s in ['train','validation','test']]
            self.assertTrue(a and b and c)
            self.assertFalse(a&b or a&c or b&c)

    def test_uncertainty_boundary_matches_game(self):
        t=dict(kind='SHORTAGE',source='{place} has little food.',target='{place} is short of food.',confidence=40,retellings=4,paraphrase=True)
        self.assertEqual(render(t,dict(place='Oakford'),0)[0],'- ~ Oakford has little food.\n')
        t['confidence']=39
        self.assertTrue(render(t,dict(place='Oakford'),0)[0].startswith('- ? ~ '))

    def test_names_require_word_boundaries(self):
        self.assertTrue(contains("Ashford's bridge",'Ashford'))
        self.assertFalse(contains('Ashfordbridge','Ashford'))

    def test_swapped_names_fail_pair(self):
        samples=[dict(id='a',generated='Bram Smith died.',expected='Ada Bell died.',stopped=True),
                 dict(id='b',generated='Bram Smith died.',expected='Bram Smith died.',stopped=True)]
        rows=[dict(id=k,required=dict(person=n),pair_id='pair',changed_field='person',facts=dict(person=n),changed_from='Ada Bell',changed_to='Bram Smith')
              for k,n in [('a','Ada Bell'),('b','Bram Smith')]]
        result=measure(samples,rows)
        self.assertEqual(result['required_names_match'],1)
        self.assertEqual(result['changed_pairs_match'],0)

    def test_builder_keeps_pairs_and_splits_intact(self):
        accounts = [
            ('NOTICE', 'Ada Bell posts a notice at Oakford: Relief charter.'),
            ('DEATH', 'Ada Bell died at age 70 after a life in Oakford.'),
            ('DRAGON FIRE', 'Ashwing burns Oakford because 17 stolen crowns remain missing.'),
            ('DRAGON OMEN', "Smoke falls into Oakford's chimneys; old readers count 14 nights until Ashwing comes."),
            ('GOBLIN RAID', 'Ash Court raids Oakford: 20 Wheat, 15 crowns.'),
            ('SHORTAGE', 'Oakford has 0 food in store. Its reserve target is 75.'),
            ('HARVEST', "Oakford's drought harvest cannot supply the southern settlements."),
            ('ROUTE', 'Oakford closes the treaty bridge and delays the relief convoy.'),
            ('PEACE', "Ash Court's courier reaches Elm Court: peace now binds the two courts."),
            ('TREASURE', 'Oakford finishes Oakford Moon Cup from 1 Raw Gold, 1 Gems, and 3 weeks of work.'),
            ('BANDITS', 'Ash Court recruits from hungry debtors and unpaid households; road control reaches 34%.'),
            ('CULT RALLIES', "Ash Court gathers 1 new tithe-bearers; Ashwing's court returns to 47."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir()
            rows=[dict(id=str(i), input=dict(kind=k,account=a,confidence=93,retellings=1),output=a)
                  for i,(k,a) in enumerate(accounts)]
            (source/'train.audit.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (source/'manifest.json').write_text('{}')
            for suffix in ['txt','audit.jsonl']:
                (source/('editorial_test.'+suffix)).write_text('fixture')
            build(source,root/'built',pairs=24)
            build(source,root/'repeat',pairs=24)
            for f in (root/'built').iterdir():
                self.assertEqual(f.read_bytes(),(root/'repeat'/f.name).read_bytes())
            for split in ['train','validation','test','wording_test']:
                examples=read_split(root/'built',split,512)
                audit=[json.loads(l) for l in (root/'built'/(split+'.audit.jsonl')).read_text().splitlines()]
                for i in range(0,len(examples),2):
                    a,b=examples[i:i+2]; ra,rb=audit[i:i+2]
                    self.assertEqual(a['prefix'].splitlines()[:-1],b['prefix'].splitlines()[:-1])
                    self.assertEqual(sum(ra['facts'][k]!=rb['facts'][k] for k in ra['facts']),1)
                    self.assertNotEqual(a['target'],b['target'])
                    for e,r in [(a,ra),(b,rb)]:
                        self.assertTrue(all(n.encode() in e['target'] and n.encode() in e['prefix'] for n in r['required'].values()))
                        self.assertEqual(r['form']==3,split=='wording_test')

    def test_batched_generation_matches_single(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(1)
        shapes=[(128,8),(8,),(8,8),(8,8),(8,8),(8,8),(8,),(16,8),(8,16),(8,)]
        model=Zero([b'ZEROLM2\0',3,128,32,8,2,1,16,10,1,0,7],
                   [rng.normal(0,.1,s).astype('float32') for s in shapes])
        examples=[dict(id=str(i),kind='NOTICE',prefix=p,target=b'News.\n') for i,p in enumerate([b'- News\n',b'- A longer event\n'])]
        self.assertEqual(generate(model,examples,'cpu',max_chars=12),generate_batch(model,examples,'cpu',max_chars=12))


if __name__=='__main__': unittest.main()
