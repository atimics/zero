"""Export a candidate.pt left behind by a training run into a .ccv2.

A run that is killed, or that ends without clearing its gate, leaves its
lowest-validation weights as `candidate.pt`. Those weights answer the question
the run was asked -- what is it actually saying now -- but only once they are
in the format the scorers read, and that the native runtime reads after
compile_model.py re-pins the file hash.

The export is named `rejected.ccv2` for the same reason the trainer names it
that: nothing that failed a gate should be one rename away from shipping.
"""
import argparse
import json
from pathlib import Path
import torch
from crownless_v2_export import load_export, export

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True, help='Training output directory')
p.add_argument('--base', type=Path, required=True, help='Model the run was tuned from')
p.add_argument('--tokenizer', type=Path, required=True)
p.add_argument('--output', type=Path, help='Defaults to <run>/rejected.ccv2')
a = p.parse_args()

held = torch.load(a.run / 'candidate.pt', map_location='cpu', weights_only=True)
model, metadata = load_export(a.base, a.tokenizer, 'cpu')
model.mode = 'conversation'
model.load_state_dict(held['state'])
output = a.output or a.run / 'rejected.ccv2'
export(model, a.tokenizer, output,
       {k: metadata[k] for k in ['meaning_ids', 'kind_ids', 'rules_sha256']} | {'step': held['step']})
print(json.dumps({'output': str(output), 'step': held['step'], 'validation': held['validation'],
                  'accepted': held['accepted'], 'guard': held.get('guard'), 'chat': held.get('chat')}))
