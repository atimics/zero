"""Build a short blind review from actual four-turn model outputs."""
import argparse,hashlib,json,random
from pathlib import Path


def packet_for(prior,expanded):
    if len(prior)!=len(expanded) or not prior:raise ValueError('Both runs need the same cases')
    rng=random.Random(915);sides=['prior','expanded']*(len(prior)//2)+(['prior'] if len(prior)%2 else []);rng.shuffle(sides)
    cases=[];key=[]
    for i,(left,right) in enumerate(zip(prior,expanded),1):
        if left['id']!=right['id'] or left['account']!=right['account']:raise ValueError('Paired accounts differ')
        if len(left['turns'])!=4 or len(right['turns'])!=4:raise ValueError('Expected four generated turns')
        prompt=left['account']
        texts={label:prompt+'\n\n'+'\n'.join(('A: ' if j%2==0 else 'B: ')+t['text'] for j,t in enumerate(row['turns'])) for label,row in [('prior',left),('expanded',right)]}
        a=sides[i-1];b='expanded' if a=='prior' else 'prior'
        cases.append(dict(case_id=i,prompt=prompt,A=texts[a],B=texts[b]));key.append(dict(case_id=i,source_id=left['id'],A=a,B=b))
    digest=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return dict(schema_version=1,packet_id='crownless-grounded-'+digest[:16],cases=cases),key


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh review directory')
    packet,key=packet_for(json.loads((a.run/'prior-world.json').read_text()),json.loads((a.run/'expanded-world.json').read_text()))
    a.output.mkdir(parents=True)
    (a.output/'ab-packet.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
    (a.output/'key.json').write_text(json.dumps(key,indent=2)+'\n')
    template=Path(__file__).with_name('ab_review.html').read_text()
    template=template.replace('ZERO · Pick the better story','Crownless · Small conversations').replace('ZERO / STORY PICKS','CROWNLESS / SMALL CONVERSATIONS').replace('Which would you keep reading?','Which conversation feels more natural?').replace('Pick A or B. Skip whenever you want.','Pick A or B. Consider how well it follows the account. Skip whenever you want.')
    payload=json.dumps(packet).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (a.output/'ab-review.html').write_text(template.replace('/*PACKET*/null',payload))
    print(packet['packet_id'])
if __name__=='__main__':main()
