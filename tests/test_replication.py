import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bootstrap_subword import bootstrap
from summarize_replication import paired
from subword_model import CONFIGS,create
class Replication(unittest.TestCase):
    def test_pairing_and_zero(self):
        rows=[{'start_token':i,'bytes':10+i,'nats':20+i} for i in range(12)]
        r=bootstrap(rows,rows,100);self.assertEqual(r['paired_window_95_percentile_interval'],[0,0]);self.assertEqual(r['ties'],12)
        with self.assertRaises(ValueError):bootstrap(rows,rows[::-1],100)
    def test_seed_interval(self):
        self.assertEqual(paired([-.02]*5)['direction'],'long_lower')
        self.assertTrue(paired([-.001,0,.001,0,0])['practical_equivalence_at_0_01'])
        self.assertFalse(paired([0])['complete'])
    def test_seed_and_parameter_controls(self):
        a=create(CONFIGS['5m-256'],11);b=create(CONFIGS['5m-1024'],11);c=create(CONFIGS['5m-256'],19)
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertFalse(torch.equal(a.weights[0],c.weights[0]))
        self.assertEqual(sum(p.numel() for p in create(CONFIGS['5m-wide-256']).parameters()),5046912)
if __name__=='__main__':unittest.main()
