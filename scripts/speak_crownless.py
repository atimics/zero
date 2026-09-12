"""Complete plain Crownless event lines with one spoken line."""
import argparse
from pathlib import Path

import torch
from evaluate_crownless_facts import generate_batch
from zero_torch import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--event',action='append',required=True,help='Repeat for older events first; final event is the topic. Prefix uncertain events with ? and widely retold events with ~.')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    args=p.parse_args()
    if any('\n' in e or '\r' in e or not e.strip() for e in args.event):
        p.error('Supply each event on one nonempty line')
    try:
        prefix=''.join('- '+e.strip()+'\n' for e in args.event).encode('ascii')
    except UnicodeEncodeError:
        p.error('This checkpoint uses ASCII text')
    torch.set_num_threads(4)
    model,_=load(args.model)
    if len(prefix)>=model.context:
        p.error('Use shorter event lines to leave room for speech')
    model.to(args.device)
    sample=generate_batch(model,[dict(id='speech',kind='user',prefix=prefix,target=b'\n')],args.device)[0]
    if not sample['stopped'] or not sample['generated']:
        p.error('The model could not finish a spoken line within the length limit')
    print(sample['generated'])


if __name__=='__main__': main()
