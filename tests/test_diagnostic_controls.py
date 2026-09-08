import json,math,sys,unittest
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from subword_diagnostics import analyze
from analyze_book_clusters import clustered,attach
ROOT=Path(__file__).resolve().parents[1]
class DiagnosticControls(unittest.TestCase):
    def test_recorded_gpu_positive_and_repaired_negative(self):
        data=json.loads((ROOT/'docs/subword/results.json').read_text());row=data['models']['5m-1024']['samples'][1]
        index=json.loads((ROOT/'experiments/subword-replication/token-placement-control.json').read_text());tok=Tokenizer.from_file(str(ROOT/'experiments/subword-scaling/tokenizer.json'))
        result=analyze([row],index,{'names':{}},tok)
        self.assertEqual(result['unusual_initial_count'],1)
        self.assertEqual(result['rows'][0]['unusual_initial_examples'][0]['text'],'irl')
        fixed={**row,'output':row['output'].replace('g irl','girl')}
        self.assertEqual(analyze([fixed],index,{'names':{}},tok)['unusual_initial_count'],0)
    def test_book_bootstrap_constant_effect(self):
        rows=[{'book_id':i//3,'bytes':20+i,'short_nats':30.,'long_nats':30.+.01*(20+i)*math.log(2)} for i in range(24)]
        interval=clustered(rows,100)['percentile_95_interval']
        for value in interval:self.assertAlmostEqual(value,.01,places=12)
    def test_cross_book_context_is_recorded(self):
        rows=[{'start_token':n,'bytes':64,'nats':20.} for n in [1200,2100]]
        ranges=[{'book_id':i,'author':str(i),'title':str(i),'start':i*2000,'end':(i+1)*2000} for i in range(2)]
        kept,excluded=attach(rows,rows,ranges,np.arange(4001));self.assertEqual(len(kept),1);self.assertEqual(len(excluded),1);self.assertTrue(excluded[0]['context_crosses_book_boundary']);self.assertFalse(excluded[0]['target_crosses_book_boundary'])
if __name__=='__main__':unittest.main()
