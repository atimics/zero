import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from grounded_dialogue import render,build,exchange_rows,replace_fields
from run_reviewed_dialogue_pilot import read,encode
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
class GroundedDialogue(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=read(ROOT/'experiments/full-event-rehearsal/input/rehearsal.jsonl')
  cls.bank=json.loads((ROOT/'experiments/grounded-dialogue/input/dialogue-bank.json').read_text())['exchanges']
  cls.meta={'meaning_ids':{k:i+1 for i,k in enumerate(cls.bank)}}
  cls.tok=Tokenizer.from_file(str(ROOT/'models/crownless-core-v2/tokenizer.json'))
 def test_utf8_repeated_fields_keep_exact_copy_spans(self):
  fields=[dict(field=0,text='Élm Vale',role=3,spoken=True,knowledge=0)]
  text,spans=render('Near {0}: {0}.',fields)
  self.assertEqual(len(spans),2)
  for span in spans:self.assertEqual(text.encode()[span['start']:span['end']].decode(),'Élm Vale')
 def test_hidden_field_stays_literal_without_copy(self):
  text,spans=render('{0}.',[dict(field=0,text='someone',role=1,spoken=True,knowledge=3)])
  self.assertEqual((text,spans),('someone.',[]))
 def test_unspoken_quantity_is_rejected(self):
  with self.assertRaises(ValueError):render('{0}',[dict(field=0,text='19',role=8,spoken=False)])
 def test_format_expression_is_rejected(self):
  with self.assertRaises(ValueError):render('{0!r}',[])
 def test_missing_meaning_fails(self):
  with self.assertRaises(ValueError):build(self.rows,{k:v for k,v in self.bank.items() if k!='harvest_failed_0'},self.meta)
 def test_all_meanings_encode_with_copy_targets(self):
  selected={}
  for row in self.rows:selected.setdefault(row['rule'],row)
  result=build(list(selected.values()),self.bank,self.meta)
  self.assertEqual(len(result),61*4)
  for row in result:
   encoded=encode(self.tok,row)
   self.assertEqual(sum(c>=0 for c in encoded['copy_targets']),len(row['copies']))
 def test_swap_changes_account_answer_and_history(self):
  source=next(r for r in self.rows if r['rule']=='bandit_pressure_1');before=copy.deepcopy(source)
  altered=replace_fields(source,{1:'The Copper Lantern'})
  turns=exchange_rows(altered,self.bank[source['rule']],self.meta)
  self.assertIn('The Copper Lantern',turns[2]['output'])
  self.assertIn('The Copper Lantern',turns[2]['history'][0]['text'])
  self.assertNotIn(source['fields'][1]['text'],turns[2]['output'])
  encoded=encode(self.tok,turns[2]);self.assertEqual([x for x in encoded['copy_targets'] if x>=0],[1])
  self.assertEqual(source,before)
 def test_uncertainty_stays_on_the_answer(self):
  source=copy.deepcopy(next(r for r in self.rows if r['rule']=='harvest_failed_0'));source['confidence']=20
  turns=exchange_rows(source,self.bank[source['rule']],self.meta)
  self.assertTrue(turns[2]['output'].endswith(', I think.'))
  encode(self.tok,turns[2])
 def test_generated_name_split_is_disjoint(self):
  test=read(ROOT/'experiments/full-event-rehearsal/input/grammar-test.jsonl')
  names=lambda rows:{f['text'] for r in rows for f in r['fields'] if f['role'] in (1,2,3,4,5) and f['knowledge']!=3}
  self.assertFalse(names(self.rows)&names(test))
if __name__=='__main__':unittest.main()
