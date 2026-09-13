"""Score one held-out grammar case per original meaning on all three models."""
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2_export import load_export
from crownless_v2 import generate
from run_reviewed_dialogue_pilot import encode,read
from score_crownless_v2 import accepted_forms

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--arms',nargs='+',choices=['base','narrow','full'],default=['base','narrow','full']);a=p.parse_args();torch.set_num_threads(4)
 root=Path('experiments/full-event-rehearsal/input');rules={r['id']:r for r in json.loads((root/'rules.json').read_text())['rules']};tok=Tokenizer.from_file('models/crownless-core-v2/tokenizer.json')
 selected={}
 for r in read(root/'grammar-test.jsonl'):selected.setdefault(r['rule'],r)
 assert set(selected)==set(rules)
 scores={}
 for label,path in [('base',Path('models/crownless-conversation/core.ccv2')),('narrow',a.run/'narrow.ccv2'),('full',a.run/'full.ccv2')]:
  if label not in a.arms:continue
  model,meta=load_export(path,'models/crownless-core-v2/tokenizer.json');model.mode='conversation';rows=[]
  for rule,r in selected.items():
   r={**r,'kind_id':meta['meaning_ids'][rule],'history':[]};g=generate(model,tok,encode(tok,r),max_tokens=100)
   rows.append(dict(id=r['id'],rule=rule,reference=r['output'],text=g['text'],stopped=g['stopped'],approved_form=g['text'] in accepted_forms(r,rules[rule]) and g['stopped']))
  (a.run/(label+'-grammar.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n');scores[label]={'count':len(rows),'approved_forms':sum(r['approved_form'] for r in rows),'stopped':sum(r['stopped'] for r in rows)}
 (a.run/'retention.json').write_text(json.dumps(scores,indent=2)+'\n');print(json.dumps(scores))
if __name__=='__main__':main()
