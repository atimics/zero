"""Fixed-budget comparison of narrow and full-grammar rehearsal."""
import argparse,collections,copy,json,random,time,re
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import batch,generate
from crownless_v2_export import load_export,export
from crownless_conversation import response
from run_reviewed_dialogue_pilot import read,sha,make_row,encode,normalized
from run_matched_dialogue_study import BASE_HASH,rows_for,select_test,rollout,metrics

def full_rehearsal(rows,rules,metadata,covered):
    if {r['rule'] for r in rows} != set(metadata['meaning_ids']):raise ValueError('Incomplete rehearsal grammar')
    rng=random.Random(914);result=[];acts=['agree','certainty','source','check','close','end','question']
    for i,source in enumerate(rows):
        r=copy.deepcopy(source);r['kind_id']=metadata['meaning_ids'][r['rule']];r['history']=[]
        result.append(r)
        if r['rule'] not in covered:
            result.append(response(r,rules[r['rule']],acts[i%len(acts)],rng))
    return result

def normalize_directions(draft, source):
    result=copy.deepcopy(draft)
    directions={'eastern':'east','southern':'south','northern':'north','western':'west'}
    for field in source['packet']['fields']:
        if field['role']!=3:continue
        for adjective,direction in directions.items():
            if field['text']!=adjective+' settlements':continue
            changed=[]
            for line in result['lines']:
                line=re.sub(r'\bsettlements to the '+direction+r'\b',field['text'],line)
                line=re.sub(r'\b([Tt])hose to the '+direction+r'\b',lambda m:('The ' if m[1]=='T' else 'the ')+field['text'],line)
                changed.append(line)
            result['lines']=changed
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);p.add_argument('--arms',nargs='+',choices=['narrow','full'],default=['narrow','full']);p.add_argument('--copy-directions',action='store_true');a=p.parse_args()
    assert a.steps>0 and not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    root=Path('experiments/full-event-rehearsal/input');prior=Path('experiments/reviewed-dialogue-pilot/input')
    bp=Path('models/crownless-conversation/core.ccv2');tp=Path('models/crownless-core-v2/tokenizer.json');assert sha(bp)==BASE_HASH
    tok=Tokenizer.from_file(str(tp));model,meta=load_export(bp,tp);model.mode='conversation'
    assert sha(root/'rules.json')==meta['rules_sha256']
    sources={r['id']:r for r in read(prior/'accounts.jsonl')};drafts={r['source_id']:r for r in json.loads((prior/'drafts.json').read_text())}
    ids=[sid for sid in drafts if sources[sid]['packet']];covered={sources[sid]['packet']['rule'] for sid in ids}
    dialogue=[r for sid in ids for r in rows_for(sources[sid],normalize_directions(drafts[sid],sources[sid]) if a.copy_directions else drafts[sid],meta)]
    old=read(root/'rehearsal.jsonl');rules={r['id']:r for r in json.loads((root/'rules.json').read_text())['rules']}
    full=full_rehearsal(old,rules,meta,covered);narrow=[make_row(sources[sid],sources[sid]['model_text'],meta) for sid in ids]
    test=select_test(read(root/'evaluation-accounts.jsonl'),[sources[i] for i in ids],16)
    assert {s['source']['provenance']['world_seed'] for s in test}.isdisjoint({sources[i]['source']['provenance']['world_seed'] for i in ids})
    rehearsal_accounts={normalized(r['prefix'].strip()) for r in full}
    assert all(normalized('- '+s['packet']['text']) not in rehearsal_accounts for s in test)
    write=lambda name,x:(a.output/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
    write('evaluation-accounts.json',test);write('dialogue.json',dialogue)
    manifest=dict(copy_directions=a.copy_directions,arms=a.arms,steps=a.steps,seed=914,parameters=sum(p.numel() for p in model.parameters()),device='cpu',torch=torch.__version__,batch_size=16,dialogue_per_batch=12,rehearsal_per_batch=4,learning_rate=.0001,dialogues=len(ids),dialogue_rows=len(dialogue),narrow_rehearsal_rows=len(narrow),full_rehearsal_rows=len(full),full_opening_rows=len(old),full_meanings=len(meta['meaning_ids']),covered_dialogue_meanings=len(covered),base_sha256=sha(bp),tokenizer_sha256=sha(tp),selection='Fixed 1200-step endpoints; same unseen test accounts for both arms. Full arm adds all-meaning openings and original conversations on uncovered meanings.',input_hashes={str(f):sha(f) for f in [root/'rehearsal.jsonl',root/'rules.json',root/'evaluation-accounts.jsonl',prior/'drafts.json',prior/'accounts.jsonl']},source_hashes={str(f):sha(f) for f in [Path(__file__),Path('scripts/run_matched_dialogue_study.py'),Path('scripts/run_reviewed_dialogue_pilot.py'),Path('scripts/crownless_conversation.py'),Path('scripts/crownless_v2.py'),Path('scripts/crownless_v2_export.py')]})
    write('manifest.json',manifest)
    before=rollout(model,tok,meta,test);write('before.json',before);summary={'before':metrics(before)};print(json.dumps(summary),flush=True)
    tr=[encode(tok,r) for r in dialogue]
    for arm in a.arms:
        replay=narrow if arm=='narrow' else full
        torch.manual_seed(914);rng=random.Random(914);model,meta=load_export(bp,tp);model.mode='conversation'
        rp=[encode(tok,r) for r in replay];optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);log=[];start=time.monotonic()
        for step in range(1,a.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True);loss=model.loss(batch(rng.choices(tr,k=12)+rng.choices(rp,k=4),'cpu'));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step==1 or step%100==0 or step==a.steps:
                item=dict(step=step,loss=loss.item(),seconds=time.monotonic()-start);log.append(item);write(arm+'-history.json',log);print(json.dumps(dict(arm=arm,**item)),flush=True)
        cp=a.output/(arm+'.ccv2');export(model,tp,cp,meta);model,meta=load_export(cp,tp);model.mode='conversation'
        after=rollout(model,tok,meta,test);write(arm+'-after.json',after)
        summary[arm]=dict(**metrics(after),opening_unchanged=sum(x['turns'][0]['text']==y['turns'][0]['text'] for x,y in zip(before,after)),candidate_sha256=sha(cp),training_seconds=log[-1]['seconds'])
        # Cover every original meaning in a fixed, untrained-name grammar test.
        # Conversation quality remains assessed from the world rollouts above.
        write('results.json',summary);print(json.dumps({arm:summary[arm]}),flush=True)
if __name__=='__main__':main()
