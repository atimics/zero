"""Measure repeated model opening inputs, with and without literal copy values."""
import argparse,json
from pathlib import Path
from tokenizers import Tokenizer
from crownless_v2_export import load_export
from run_reviewed_dialogue_pilot import encode,make_row

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
 tok=Tokenizer.from_file('models/crownless-core-v2/tokenizer.json')
 _,meta=load_export('models/crownless-conversation/core.ccv2','models/crownless-core-v2/tokenizer.json')
 rows=json.loads((a.run/'broad-train.json').read_text())+json.loads((a.run/'broad-replay.json').read_text())
 tests=json.loads((a.run/'evaluation-accounts.json').read_text())
 def sig(r,values=False):
  x=encode(tok,r);n=x['prefix_length'];parts=[x['tokens'][:n],x['meta'][:n]]
  if values:parts.append(x['source_ids'])
  return json.dumps(parts)
 slots={sig(r) for r in rows};full={sig(r,True) for r in rows};cases=[]
 for s in tests:
  r=make_row(s,'',meta,own=True);cases.append(dict(id=s['id'],slot_prefix_seen=sig(r) in slots,prefix_and_copy_values_seen=sig(r,True) in full))
 report=dict(cases=cases,slot_prefix_seen=sum(x['slot_prefix_seen'] for x in cases),prefix_and_copy_values_seen=sum(x['prefix_and_copy_values_seen'] for x in cases),scope='Opening inputs only. Repeated slot patterns with fresh copy values measure entity transfer inside learned account patterns.')
 (a.run/'input-overlap.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
