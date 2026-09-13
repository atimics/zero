import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_matched_dialogue_study import full_names, rows_for, select_test

class MatchedStudy(unittest.TestCase):
    def setUp(self):
        self.source={'id':'x','packet':{'text':'Alda Millrace joined.','rule':'r','fields':[{'start':0,'end':13,'text':'Alda Millrace','field':0,'spoken':True,'knowledge':0,'role':1,'provenance':3,'event':1}]},'model_text':'Alda Millrace joined.','event':{'kind':1},'source':{'input':{'account':'Alda Millrace joined.','confidence':60,'retellings':5}}}
    def test_partial_name_uses_existing_full_name(self):
        self.assertEqual(full_names('Alda? Alda Millrace?',self.source),'Alda Millrace? Alda Millrace?')
    def test_both_speakers_keep_fields_and_relative_history(self):
        rows=rows_for(self.source,{'lines':['Alda Millrace has joined.','Alda?','Yes.']},{'meaning_ids':{'r':1}})
        self.assertEqual(len(rows),6)
        self.assertTrue(all(r['kind_id']==1 and r['fields'] for r in rows))
        self.assertEqual(rows[1]['history'][0]['speaker'],'other')
        self.assertEqual([h['speaker'] for h in rows[2]['history']],['self','other'])
        self.assertEqual(rows[4]['history'][0]['text'],self.source['model_text'])
    def test_equal_openings_are_deduplicated(self):
        rows=rows_for(self.source,{'lines':[self.source['model_text'],'Yes.']},{'meaning_ids':{'r':1}})
        self.assertEqual(len(rows),2)
    def test_test_excludes_matching_normalized_account(self):
        train=copy.deepcopy(self.source);train['source']['input']['account']='Town made 4 bread.'
        a=copy.deepcopy(train);a['source']['input']['account']='TOWN made 20 bread.'
        b=copy.deepcopy(train);b['id']='b';b['source']['input']['account']='Another town made bread.'
        self.assertEqual([r['id'] for r in select_test([a,b],[train],1)],['b'])
if __name__=='__main__':unittest.main()
