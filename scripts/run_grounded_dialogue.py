"""Fixed-budget full-coverage dialogue comparison with explicit copy targets."""
import argparse,copy,json,random,time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import batch,generate
from crownless_v2_export import load_export,export
from grounded_dialogue import build,exchange_rows,replace_fields
from run_reviewed_dialogue_pilot import read,sha,encode,make_row
from run_matched_dialogue_study import BASE_HASH,rows_for,select_test,rollout,metrics
from run_full_event_rehearsal import full_rehearsal
from score_crownless_v2 import accepted_forms


def diagnostics(model,tok,meta,bank,testrows,rules):
    selected={}
    for row in testrows:selected.setdefault(row['rule'],row)
    assert set(selected)==set(bank)
    openings=[];answers=[]
    for rule,source in selected.items():
        opening=exchange_rows(source,bank[rule],meta)[0]
        g=generate(model,tok,encode(tok,opening),max_tokens=100)
        openings.append(dict(rule=rule,reference=source['output'],**g,accepted=g['stopped'] and g['text'] in accepted_forms(source,rules[rule])))
        expected=exchange_rows(source,bank[rule],meta)[2]
        cases=[('original',source)]
        # Change each named answer field separately. Material and detail values
        # stay within their own grammar rather than becoming place names.
        slots={f['field'] for f in expected['copies'] if f['role'] in (1,2,3,4,5)}
        for slot in sorted(slots):
            for name in ['Élm Vale','The Copper Lantern']:
                cases.append((f'field-{slot}:{name}',replace_fields(source,{slot:name})))
        for label,case in cases:
            target=exchange_rows(case,bank[rule],meta)[2];record=encode(tok,target)
            g=generate(model,tok,record,max_tokens=80)
            wanted=[v for v in record['copy_targets'] if v>=0]
            actual=[a['copy'] for a in g['actions'] if 'copy' in a]
            answers.append(dict(rule=rule,case=label,confidence=case['confidence'],retold=case['retold'],history=target['history'],reference=target['output'],**g,expected_copies=wanted,actual_copies=actual,copy_exact=wanted==actual,exact=g['stopped'] and g['text']==target['output']))
    return dict(openings=openings,answers=answers,summary=dict(openings_accepted=sum(x['accepted'] for x in openings),openings=len(openings),answers_exact=sum(x['exact'] for x in answers),answers=len(answers),copy_cases=sum(bool(x['expected_copies']) for x in answers),copy_cases_exact=sum(bool(x['expected_copies']) and x['copy_exact'] and x['stopped'] for x in answers)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2400);a=p.parse_args()
    if a.steps<=0 or a.output.exists():raise ValueError('Use positive steps and a fresh output directory')
    a.output.mkdir(parents=True);torch.set_num_threads(4)
    root=Path('experiments/grounded-dialogue/input');oldroot=Path('experiments/full-event-rehearsal/input');prior=Path('experiments/reviewed-dialogue-pilot/input')
    bp=Path('models/crownless-conversation/core.ccv2');tp=Path('models/crownless-core-v2/tokenizer.json');assert sha(bp)==BASE_HASH
    tok=Tokenizer.from_file(str(tp));model,meta=load_export(bp,tp);model.mode='conversation'
    bank=json.loads((root/'dialogue-bank.json').read_text())['exchanges']
    old=read(oldroot/'rehearsal.jsonl');rules={r['id']:r for r in json.loads((oldroot/'rules.json').read_text())['rules']};assert sha(oldroot/'rules.json')==meta['rules_sha256']
    sources={r['id']:r for r in read(prior/'accounts.jsonl')};drafts={r['source_id']:r for r in json.loads((prior/'drafts.json').read_text())};ids=[i for i in drafts if sources[i]['packet']]
    original=[r for i in ids for r in rows_for(sources[i],drafts[i],meta)]
    covered={sources[i]['packet']['rule'] for i in ids}
    training={'prior':original,'expanded':build(old,bank,meta)}
    rehearsal={'prior':full_rehearsal(old,rules,meta,covered),'expanded':full_rehearsal(old,rules,meta,set(rules))}
    test=select_test(read(root/'evaluation-accounts.jsonl'),[sources[i] for i in ids],16)
    assert {s['source']['provenance']['world_seed'] for s in test}=={1901,1902}
    train_names={f['text'] for r in old for f in r['fields'] if f['role'] in (1,2,3,4,5) and f['knowledge']!=3}
    testrows=read(oldroot/'grammar-test.jsonl')
    test_names={f['text'] for r in testrows for f in r['fields'] if f['role'] in (1,2,3,4,5) and f['knowledge']!=3}
    assert train_names.isdisjoint(test_names)
    # Encoder checks byte spans, copy markers, and the runtime context limit.
    encoded={arm:[encode(tok,r) for r in rows] for arm,rows in training.items()}
    replay={arm:[encode(tok,r) for r in rows] for arm,rows in rehearsal.items()}
    write=lambda name,x:(a.output/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
    inputs=[root/'dialogue-bank.json',root/'evaluation-accounts.jsonl',oldroot/'rehearsal.jsonl',oldroot/'grammar-test.jsonl',oldroot/'rules.json',prior/'accounts.jsonl',prior/'drafts.json']
    code=[Path(__file__),Path('scripts/grounded_dialogue.py'),Path('scripts/run_reviewed_dialogue_pilot.py'),Path('scripts/run_matched_dialogue_study.py'),Path('scripts/run_full_event_rehearsal.py'),Path('scripts/crownless_v2.py'),Path('scripts/crownless_v2_export.py')]
    write('manifest.json',dict(seed=915,steps=a.steps,selection='Fixed final checkpoint per arm',parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,device='cpu',threads=4,batch=16,dialogue_per_batch=12,rehearsal_per_batch=4,learning_rate=.0001,training_rows={k:len(v) for k,v in training.items()},rehearsal_rows={k:len(v) for k,v in rehearsal.items()},input_hashes={str(p):sha(p) for p in inputs},source_hashes={str(p):sha(p) for p in code},base_sha256=sha(bp),tokenizer_sha256=sha(tp),copy_split='Complete generated names are disjoint. Shared meaning grammar and question templates.',authorship='Assistant-authored expansion; human review pending. Prior arm includes 12 human-approved scenes out of 46.'))
    write('test-accounts.json',test)
    # The compact recipe and pinned inputs reproduce all 14,640 training rows.
    examples={}
    for r in training['expanded']:examples.setdefault(r['rule'],[])
    for rule in examples:
        first=next(r for r in old if r['rule']==rule);examples[rule]=exchange_rows(first,bank[rule],meta)
    write('training-examples.json',examples)
    before=rollout(model,tok,meta,test);write('base-world.json',before);summary={'base':metrics(before)}
    for arm in training:
        torch.manual_seed(915);rng=random.Random(915);model,meta=load_export(bp,tp);model.mode='conversation'
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);log=[];start=time.monotonic()
        for step in range(1,a.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=model.loss(batch(rng.choices(encoded[arm],k=12)+rng.choices(replay[arm],k=4),'cpu'));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step==1 or step%200==0 or step==a.steps:
                item=dict(step=step,loss=loss.item(),seconds=time.monotonic()-start);log.append(item);write(arm+'-history.json',log);print(json.dumps(dict(arm=arm,**item)),flush=True)
        path=a.output/(arm+'.ccv2');export(model,tp,path,meta);model,meta=load_export(path,tp);model.mode='conversation'
        world=rollout(model,tok,meta,test);write(arm+'-world.json',world)
        check=diagnostics(model,tok,meta,bank,testrows,rules);write(arm+'-diagnostics.json',check)
        summary[arm]=dict(world=metrics(world),diagnostics=check['summary'],candidate_sha256=sha(path),training_seconds=log[-1]['seconds']);write('results.json',summary);print(json.dumps({arm:summary[arm]}),flush=True)
if __name__=='__main__':main()
