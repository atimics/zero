"""Check fresh question wording against the fixed final candidates."""
import argparse,json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import generate
from crownless_v2_export import load_export
from grounded_dialogue import exchange_rows
from run_reviewed_dialogue_pilot import read,encode,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--contrasts',action='store_true');a=p.parse_args();torch.set_num_threads(4)
 root=Path('experiments/grounded-dialogue/input');name='question-contrasts.json' if a.contrasts else 'question-probes.json';qp=root/name
 questions=json.loads(qp.read_text())['questions'];bank=json.loads((root/'dialogue-bank.json').read_text())['exchanges'];selected={}
 for row in read('experiments/full-event-rehearsal/input/grammar-test.jsonl'):
  if row['rule'] in questions:selected.setdefault(row['rule'],row)
 assert set(selected)==set(questions)
 tok=Tokenizer.from_file('models/crownless-core-v2/tokenizer.json');results={}
 for arm in ['prior','expanded']:
  model,meta=load_export(a.run/(arm+'.ccv2'),'models/crownless-core-v2/tokenizer.json');model.mode='conversation';answers=[]
  for rule,row in selected.items():
   question=questions[rule][0] if a.contrasts else questions[rule]
   templates=[question,questions[rule][1],bank[rule][2]] if a.contrasts else bank[rule]
   target=exchange_rows(row,templates,meta)[2];target['history'][-1]['text']=question
   record=encode(tok,target);g=generate(model,tok,record,max_tokens=80)
   wanted=[c for c in record['copy_targets'] if c>=0];actual=[x['copy'] for x in g['actions'] if 'copy' in x]
   answers.append(dict(rule=rule,question=question,reference=target['output'],**g,expected_copies=wanted,actual_copies=actual,copy_exact=wanted==actual,exact=g['stopped'] and g['text']==target['output']))
  results[arm]=answers
 result=dict(probe_sha256=sha(qp),source_sha256=sha(Path(__file__)),answers=results)
 (a.run/name).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({arm:dict(exact=sum(x['exact'] for x in rows),copies=sum(x['copy_exact'] for x in rows),count=len(rows)) for arm,rows in results.items()}))
if __name__=='__main__':main()
