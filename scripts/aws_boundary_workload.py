"""Preflight, diagnose and train the bounded four-arm boundary pilot."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from boundary_common import contract, write, EXPERIMENT, identity
from boundary_data import forward, load_records, roster, roster_digest, packs
from canada_narrative import prepare, read_json, digest
from run_canada_narrative import setup, amp, synchronize, evaluate
from train_boundaries import train
from diagnose_boundaries import diagnose, mechanism_check


def preflight(data_dir, output):
    config=contract();device='cuda'
    if not torch.cuda.is_available():raise ValueError('Cloud worker requires CUDA')
    data=np.memmap(data_dir/'B.bin',dtype='<u2',mode='r')
    records=load_records(data_dir/'B.records.jsonl',len(data))
    model,optimizer=setup(config['seed'],device)
    visits=roster(records,config['target_presentations'])
    expected=roster_digest(visits,canonical=True)
    if expected!=roster_digest(roster(records,config['target_presentations'],True),canonical=True):
        raise ValueError('Target rosters differ')
    rows=list(__import__('itertools').islice(packs(data,records,visits,model.context),4))
    x,y,segments=[torch.from_numpy(np.stack([r[i] for r in rows])).to(device) for i in range(3)]
    checks=mechanism_check(model,x[:1],device);timings={}
    for isolate in [False,True]:
        times=[]
        for iteration in range(4):
            model.train();optimizer.zero_grad(set_to_none=True);synchronize(device);start=time.monotonic()
            with amp(device):
                logits=forward(model,x,segments,isolate,.1)
                loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),y.flatten())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();synchronize(device)
            if iteration:times.append(time.monotonic()-start)
        timings[str(isolate)]=float(np.median(times))*2
    steps=(sum(n+1 for _,n in visits)+model.context*8-1)//(model.context*8)
    projected_training=2*steps*(timings['False']+timings['True'])
    result={'status':'passed','mechanism':checks,'target_multiset_sha256':expected,
            'seconds_per_update':timings,'four_arm_training_seconds':projected_training,
            'planning_seconds_with_evaluation_margin':projected_training*1.4+2700}
    write(output/'preflight.json',result)
    if result['planning_seconds_with_evaluation_margin']>config['cloud']['worker_seconds']-900:
        raise ValueError('Measured workload exceeds the bounded worker envelope')
    return result


def review_packets(output, results):
    import hashlib,json
    from canada_narrative import ROOT
    baseline=results['P0']
    for arm in ['P1','P2','P3']:
        candidate=results[arm]
        if [r['case_id'] for r in baseline]!=[r['case_id'] for r in candidate]:
            raise ValueError('Sample cases differ across arms')
        order=sorted(range(len(baseline)),key=lambda i:hashlib.sha256(f'boundary-v1:{arm}:{i}'.encode()).digest())
        left=set(order[:len(order)//2]);cases=[];key=[]
        for i,(a,b) in enumerate(zip(baseline,candidate)):
            if a['prompt']!=b['prompt']:raise ValueError('Paired prompts differ')
            cases.append({'case_id':i+1,'prompt':a['prompt'],
                          'A':(b if i in left else a)['continuation'],
                          'B':(a if i in left else b)['continuation']})
            key.append({'case_id':i+1,'source_case_id':a['case_id'],'family':a['family'],
                        'candidate':'A' if i in left else 'B'})
        packet={'schema_version':1,'packet_id':'boundary-'+identity(cases)[:16],'cases':cases}
        directory=output/('P0-'+arm);write(directory/'ab-packet.json',packet);write(directory/'blind-key.json',key)
        payload=json.dumps(packet).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        html=(ROOT/'scripts/ab_review.html').read_text().replace('/*PACKET*/null',payload)
        (directory/'ab-review.html').write_text(html)


def run(delivery,data,anchor,output,preflight_only=False):
    config=contract();output.mkdir(parents=True,exist_ok=True)
    if not (data/'manifest.json').exists():prepare(delivery,data)
    else:
        from canada_narrative import verify_prepared
        verify_prepared(data)
    # The same accepted preparation remains a bound input to the new experiment.
    from canada_narrative import EXPERIMENT as CANADA
    if digest(data/'manifest.json')!=read_json(CANADA/'preparation.json')['manifest_sha256']:
        raise ValueError('Prepared streams differ from the accepted receipt')
    preflight(data,output)
    if preflight_only:return
    anchor_result=read_json(anchor/'result.json')
    if anchor_result['comparison']!='BC' or anchor_result['arm']!='B' or digest(anchor/'best.pt')!=anchor_result['best_sha256']:
        raise ValueError('Development anchor must be the completed BC-B checkpoint')
    extra=EXPERIMENT/'development-scenes.json'
    diagnose(data,anchor/'best.pt',output/'anchor-diagnostics',extra_cases=extra)
    validation=np.memmap(data/'validation.bin',dtype='<u2',mode='r')
    lengths=np.array(read_json(data/'token_bytes.json'));evaluation=read_json(data/'evaluation.json')
    scores={};samples={};runs={}
    for arm in config['arms']:
        runs[arm]=train(data,output/arm,arm)
        model,optimizer=setup(config['seed'],'cuda');del optimizer
        state=torch.load(output/arm/'best.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(state['model'])
        scores[arm]=evaluate(model,validation,evaluation['outcome'],lengths,'cuda')
        del model,state
        samples[arm]=diagnose(data,output/arm/'best.pt',output/(arm+'-samples'),['registered'],extra)
    review_packets(output,samples)
    b={a:s['bits_per_byte'] for a,s in scores.items()}
    write(output/'boundary-result.json',{'status':'passed','meaning':'Exploratory four-arm execution completed',
          'scores':scores,'runs':runs,'human_review':'pending','promotion':'requires fresh evaluation',
          'factorial_bpb_effects':{'shuffle':((b['P1']-b['P0'])+(b['P3']-b['P2']))/2,
                                  'isolation':((b['P2']-b['P0'])+(b['P3']-b['P1']))/2,
                                  'interaction':b['P3']-b['P2']-b['P1']+b['P0']},
          'effect_direction':'Negative bits-per-byte effects indicate lower development loss'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['delivery','data','anchor','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    a=parser.parse_args();run(a.delivery,a.data,a.anchor,a.output,a.preflight_only)
