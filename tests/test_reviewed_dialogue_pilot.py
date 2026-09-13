"""Knowledge isolation and byte-accurate copying for reviewed conversations."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_reviewed_dialogue_pilot import make_row, normalized

class Rows(unittest.TestCase):
    def setUp(self):
        self.s={'id':'x','packet':{'text':'Town baked bread.','rule':'bread','fields':[{'start':0,'end':4,'text':'Town','field':0,'spoken':True,'knowledge':0,'role':3,'provenance':3,'event':1}]},'source':{'input':{'confidence':30,'retellings':7}}}
        self.meta={'meaning_ids':{'bread':1}}
    def test_listener_has_only_history(self):
        h=[{'speaker':'other','text':'Town has bread.'}]
        r=make_row(self.s,'What sort?',self.meta,h,own=False)
        self.assertEqual((r['prefix'],r['fields'],r['copies'],r['kind_id']),('',[],[],0))
        self.assertEqual(r['history'],h)
    def test_copy_uses_bytes_after_unicode(self):
        r=make_row(self.s,'“Town”',self.meta)
        c=r['copies'][0]
        self.assertEqual(r['output'].encode()[c['start']:c['end']].decode(),'Town')
        self.assertTrue(r['retold']);self.assertEqual(r['confidence'],30)
    def test_partial_name_is_not_copy(self):
        self.assertEqual(make_row(self.s,'Township',self.meta)['copies'],[])
    def test_numeral_only_changes_share_split_key(self):
        self.assertEqual(normalized('Town made 4 bread.'),normalized('TOWN made 20 bread.'))
if __name__=='__main__':unittest.main()
