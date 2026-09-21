import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_grounded_review import packet_for
class GroundedReview(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(id=str(i),account='Account '+str(i),turns=[dict(text='line '+str(j)) for j in range(4)]) for i in range(16)]
 def test_balanced_stable_and_exact_output(self):
  packet,key=packet_for(self.rows,self.rows)
  self.assertEqual((packet,key),packet_for(self.rows,self.rows))
  self.assertEqual(sum(k['A']=='expanded' for k in key),8)
  for case in packet['cases']:self.assertTrue(case['A'].endswith('A: line 0\nB: line 1\nA: line 2\nB: line 3'))
 def test_mismatched_account_fails(self):
  other=copy.deepcopy(self.rows);other[0]['account']='Different'
  with self.assertRaises(ValueError):packet_for(self.rows,other)
if __name__=='__main__':unittest.main()
