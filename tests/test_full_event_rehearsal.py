import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_full_event_rehearsal import full_rehearsal,normalize_directions
from run_reviewed_dialogue_pilot import read
import json
ROOT=Path(__file__).resolve().parents[1]/'experiments/full-event-rehearsal/input'
class FullRehearsal(unittest.TestCase):
 def setUp(self):
  self.rows=read(ROOT/'rehearsal.jsonl');self.rules={r['id']:r for r in json.loads((ROOT/'rules.json').read_text())['rules']};self.meta={'meaning_ids':{k:i+1 for i,k in enumerate(self.rules)}}
 def test_all_meanings_keep_openings(self):
  result=full_rehearsal(self.rows,self.rules,self.meta,set())
  self.assertEqual({r['rule'] for r in result if not r['history']},set(self.rules))
  self.assertEqual(sum(not r['history'] for r in result),len(self.rows))
 def test_covered_meanings_skip_old_conversation(self):
  result=full_rehearsal(self.rows,self.rules,self.meta,set(self.rules))
  self.assertEqual(len(result),len(self.rows));self.assertTrue(all(not r['history'] for r in result))
 def test_missing_meaning_fails(self):
  absent=self.rows[0]['rule']
  with self.assertRaises(ValueError):full_rehearsal([r for r in self.rows if r['rule']!=absent],self.rules,self.meta,set())
 def test_builder_leaves_source_unchanged(self):
  old=copy.deepcopy(self.rows[:]);full_rehearsal(self.rows,self.rules,self.meta,set());self.assertEqual(self.rows,old)
class DirectionCopy(unittest.TestCase):
 def test_direction_phrases_match_copy_field(self):
  for adjective,direction in [('southern','south'),('eastern','east'),('western','west'),('northern','north')]:
   source={'packet':{'fields':[{'role':3,'text':adjective+' settlements'}]}}
   draft={'lines':['Those to the '+direction+'.','The settlements to the '+direction+'.']}
   before=copy.deepcopy((source,draft));result=normalize_directions(draft,source)
   self.assertEqual(result['lines'],['The '+adjective+' settlements.']*2)
   self.assertEqual((source,draft),before)
 def test_normalized_destination_gets_copy_target(self):
  from run_reviewed_dialogue_pilot import make_row
  source={'id':'test','packet':{'rule':'harvest','text':'account','fields':[{'role':3,'text':'southern settlements','spoken':True,'knowledge':1}]},'source':{'input':{'confidence':80,'retellings':0}}}
  line=normalize_directions({'lines':['Those to the south.']},source)['lines'][0]
  row=make_row(source,line,{'meaning_ids':{'harvest':1}})
  self.assertEqual([f['text'] for f in row['copies']],['southern settlements'])
 def test_other_places_keep_words(self):
  source={'packet':{'fields':[{'role':3,'text':'Thornford'}]}}
  draft={'lines':['Those to the south.']}
  self.assertEqual(normalize_directions(draft,source),draft)
if __name__=='__main__':unittest.main()
