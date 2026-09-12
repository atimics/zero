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
from zero_cached import CachedZero
from build_crownless_mixed import mixed, wide_pools, add_strict_splits


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

    def test_strict_names_exclude_names_inside_other_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'train.txt').write_text('The Ashford Cup was made in Elmfordbridge.')
            raw=bytearray();rows=[]
            for i,name in enumerate(['Ashford','Oakford','Elmford','Elmhaven']):
                prefix='- '+name+' has little food.\n';target=name+' was short of food.'
                rows.append(dict(id=str(i),pair_id=str(i//2),facts=dict(place=name),input=dict(kind='SHORTAGE'),
                                 output=target,text_start=len(raw),output_start=len(raw)+len(prefix),text_bytes=len(prefix+target+'\n\n')))
                raw.extend((prefix+target+'\n\n').encode())
            for split in ['validation','test','wording_test']:
                (root/(split+'.txt')).write_bytes(raw)
                (root/(split+'.audit.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows))
            checks=add_strict_splits(root)
            self.assertEqual(checks['strict_test']['rows'],2)
            self.assertEqual(checks['strict_test']['excluded_pairs'][0]['names'],['Ashford'])
            self.assertEqual([e['id'] for e in read_split(root,'strict_test',512)],['2','3'])

    def test_original_game_scoring_respects_omitted_actor(self):
        row=dict(id='old',input=dict(kind='NOTICE',account='Ada Bell posts a notice at Oakford: Relief charter.'),
                 output='Someone put up a relief charter in Oakford.')
        sample=dict(id='old',generated=row['output'],expected=row['output'],stopped=True)
        scores=measure([sample],[row])
        self.assertEqual(sample['required'],dict(place='Oakford'))
        self.assertEqual(scores['required_names_match'],1)

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
            raw=bytearray()
            for r in rows:
                prefix='- '+r['input']['account']+'\n'
                r.update(text_start=len(raw),output_start=len(raw)+len(prefix))
                sample=(prefix+r['output']+'\n\n').encode()
                r['text_bytes']=len(sample); raw.extend(sample)
            (source/'train.txt').write_bytes(raw)
            (source/'train.audit.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (source/'manifest.json').write_text('{}')
            for suffix in ['txt','audit.jsonl']:
                (source/('editorial_test.'+suffix)).write_text('fixture')
            build(source,root/'built',pairs=24)
            build(source,root/'repeat',pairs=24)
            for f in (root/'built').iterdir():
                self.assertEqual(f.read_bytes(),(root/'repeat'/f.name).read_bytes())
            manifest=mixed(source,root/'mixed',pairs=24)
            mixed_train=read_split(root/'mixed','train',512)
            self.assertEqual(len(mixed_train),48+2*len(rows))
            self.assertEqual(manifest['replay_rows'],2*len(rows))
            self.assertEqual(mixed_train[-1]['target'],(rows[-1]['output']+'\n').encode())
            ps,known=wide_pools(source,73)
            for role,names in known.items():
                self.assertTrue(names<=set(ps['train'][role]))
                self.assertFalse(names&set(ps['test'][role]))
                self.assertFalse(set(ps['train'][role])&set(ps['validation'][role]))
            self.assertTrue(any(n.startswith('The ') for n in ps['train']['faction']))
            self.assertTrue(any(' the ' in n for n in ps['train']['dragon']))
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
        sequences=[list(e['prefix']) for e in examples]
        tokens=torch.zeros((2,max(map(len,sequences))),dtype=torch.long)
        for i,seq in enumerate(sequences): tokens[i,:len(seq)]=torch.tensor(seq)
        with torch.no_grad():
            cache=CachedZero(model)
            logits=cache.prefill(tokens,torch.tensor(list(map(len,sequences))))
            for _ in range(5):
                for i,seq in enumerate(sequences):
                    torch.testing.assert_close(logits[i],model(torch.tensor([seq]))[0,-1],atol=1e-6,rtol=1e-5)
                new=logits.argmax(-1)
                for i,token in enumerate(new.tolist()): sequences[i].append(token)
                logits=cache.step(new)


if __name__=='__main__': unittest.main()
