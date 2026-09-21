"""Read four turns for the six opening differences observed in the initial study."""
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import generate
from crownless_v2_export import load_export
from run_reviewed_dialogue_pilot import read,encode
RULES={'bakery_production_0','paper_milled_0','masonry_repair_0','cow_slaughtered_0','harvest_failed_0','dragon_retaliation_0'}
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();torch.set_num_threads(4)
 model,meta=load_export(a.run/'full.ccv2','models/crownless-core-v2/tokenizer.json');model.mode='conversation';tok=Tokenizer.from_file('models/crownless-core-v2/tokenizer.json')
 selected={}
 for row in read('experiments/full-event-rehearsal/input/grammar-test.jsonl'):
  if row['rule'] in RULES:selected.setdefault(row['rule'],row)
 results=[]
 for rule,row in selected.items():
  turns=[]
  for t in range(4):
   history=[dict(speaker='self' if j%2==t%2 else 'other',text=line) for j,line in enumerate(turns)][-4:]
   r={**row,'history':history,'kind_id':meta['meaning_ids'][rule]}
   turns.append(generate(model,tok,encode(tok,r),max_tokens=80)['text'])
  results.append(dict(rule=rule,reference=row['output'],turns=turns))
 (a.run/'opening-differences-followup.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':main()
