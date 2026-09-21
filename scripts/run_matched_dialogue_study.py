"""Equal-budget dialogue tuning with held accounts on both sides."""
import argparse, collections, hashlib, json, random, re, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import batch, generate
from crownless_v2_export import load_export, export
from run_reviewed_dialogue_pilot import APPROVED, make_row, encode, read, normalized, sha

BASE_HASH='244a809aba71679efe129495bb981ca8d7fbe0d9012948a5f88731dabcb9c48b'

def full_names(text, source):
    """Keep a person's partial first-name mentions tied to the copy field."""
    names=[f['text'] for f in source['packet']['fields'] if f['role']==1 and f.get('spoken') and len(f['text'].split())>1 and f['text'].split()[0] not in ('A','The')]
    aliases={}
    for n in names:
        first=n.split()[0]
        if sum(x.split()[0]==first for x in names)==1:aliases[first]=n
    mappings={**aliases,**{n:n for n in names}}
    if not mappings:return text
    pattern=r'(?<!\w)(?:'+'|'.join(re.escape(n) for n in sorted(mappings,key=len,reverse=True))+r')(?!\w)'
    return re.sub(pattern,lambda m:mappings[m.group()],text)

def rows_for(source, draft, metadata):
    lines=[full_names(l,source) for l in draft['lines']]
    openings=list(dict.fromkeys([lines[0],source['model_text']]))
    rows=[]
    for opening in openings:
        sequence=[opening]+lines[1:]
        for t,line in enumerate(sequence):
            history=[dict(speaker='self' if j%2==t%2 else 'other',text=l) for j,l in enumerate(sequence[:t])][-4:]
            rows.append(make_row(source,line,metadata,history,own=True))
    return rows

def select_test(sources, training_sources, count):
    blocked={normalized(s['source']['input']['account']) for s in training_sources}
    pools=collections.defaultdict(list);seen=set()
    for s in sources:
        if not s['packet']:continue
        key=normalized(s['source']['input']['account'])
        if key in blocked or key in seen:continue
        seen.add(key);pools[s['event']['kind']].append(s)
    result=[]
    while len(result)<count and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(result)<count:result.append(pools[k].pop(0))
    if len(result)<count:raise ValueError('Too few separate test accounts')
    return result

def rollout(model,tok,meta,sources):
    result=[]
    for source in sources:
        history=[];turns=[]
        for t in range(4):
            relative=[dict(speaker='self' if j%2==t%2 else 'other',text=l) for j,l in enumerate(history)][-4:]
            r=make_row(source,'',meta,relative,own=True)
            out=generate(model,tok,encode(tok,r),max_tokens=80)
            turns.append(dict(text=out['text'],stopped=out['stopped']));history.append(out['text'])
        result.append(dict(id=source['id'],account=source['source']['input']['account'],confidence=source['source']['input']['confidence'],turns=turns))
    return result

def metrics(rows):
    lines=[t['text'] for r in rows for t in r['turns']]
    return dict(lines=len(lines),distinct=len(set(lines)),stopped=sum(t['stopped'] for r in rows for t in r['turns']),empty=sum(not l for l in lines),replacement_character=sum('\ufffd' in l for l in lines))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);p.add_argument('--test-count',type=int,default=16)
    a=p.parse_args();assert a.steps>0 and a.test_count>0 and not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    base=Path('models/crownless-conversation/core.ccv2');tp=Path('models/crownless-core-v2/tokenizer.json');assert sha(base)==BASE_HASH
    tok=Tokenizer.from_file(str(tp));_,metadata=load_export(base,tp)
    inp=Path('experiments/reviewed-dialogue-pilot/input')
    sources={r['id']:r for r in read(inp/'accounts.jsonl')};drafts={r['source_id']:r for r in json.loads((inp/'drafts.json').read_text())}
    broad=[sid for sid in drafts if sources[sid]['packet']]
    testpath=Path('experiments/matched-dialogue-study/input/evaluation-accounts.jsonl')
    test=select_test(read(testpath),[sources[s] for s in broad],a.test_count)
    assert {s['source']['provenance']['world_seed'] for s in test}.isdisjoint({sources[s]['source']['provenance']['world_seed'] for s in broad})
    write=lambda name,obj:(a.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    write('evaluation-accounts.json',test)
    # Freeze both training sets and evaluation before any model generation.
    training={name:[r for sid in ids for r in rows_for(sources[sid],drafts[sid],metadata)] for name,ids in [('small',APPROVED),('broad',broad)]}
    replay={name:[make_row(sources[sid],sources[sid]['model_text'],metadata) for sid in ids] for name,ids in [('small',APPROVED),('broad',broad)]}
    for name in training:
        write(name+'-train.json',training[name]);write(name+'-replay.json',replay[name])
    info=dict(steps=a.steps,seed=913,device='cpu',batch_size=16,dialogue_per_batch=12,replay_per_batch=4,learning_rate=.0001,small_dialogues=len(APPROVED),broad_dialogues=len(broad),training_rows={k:len(v) for k,v in training.items()},excluded_sources=[sid for sid in drafts if sid not in broad],both_speakers_hold_same_account=True,parameters=4935937,torch=torch.__version__,base_sha256=sha(base),tokenizer_sha256=sha(tp),source_hashes={str(f):sha(f) for f in [Path(__file__),Path('scripts/run_reviewed_dialogue_pilot.py'),Path('scripts/crownless_v2.py'),Path('scripts/crownless_v2_export.py'),inp/'accounts.jsonl',inp/'drafts.json',testpath]},scope='Equal-budget local study. Twelve approved scenes versus 46 usable candidate scenes. Original and model-opening histories; full-name copy normalization. Fixed final checkpoints; shared event grammar remains.',test_ids=[s['id'] for s in test])
    write('manifest.json',info)
    model,meta=load_export(base,tp);model.mode='conversation';before=rollout(model,tok,meta,test);write('before.json',before)
    summary={'before':metrics(before)};print(json.dumps(summary),flush=True)
    for name in ['small','broad']:
        torch.manual_seed(913);rng=random.Random(913);model,meta=load_export(base,tp);model.mode='conversation'
        rows=[encode(tok,r) for r in training[name]];rp=[encode(tok,r) for r in replay[name]]
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);start=time.monotonic();log=[]
        for step in range(1,a.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=model.loss(batch(rng.choices(rows,k=12)+rng.choices(rp,k=4),'cpu'));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step==1 or step%100==0 or step==a.steps:
                item=dict(step=step,loss=loss.item(),seconds=time.monotonic()-start);log.append(item);write(name+'-history.json',log);print(json.dumps(dict(arm=name,**item)),flush=True)
        candidate=a.output/(name+'.ccv2');export(model,tp,candidate,meta);model,meta=load_export(candidate,tp);model.mode='conversation'
        after=rollout(model,tok,meta,test);write(name+'-after.json',after)
        fit=[]
        for r in training[name][::2]:
            out=generate(model,tok,encode(tok,r),max_tokens=80);fit.append(dict(id=r['id'],reference=r['output'],history=r['history'],text=out['text'],exact=out['text']==r['output']))
        write(name+'-fit.json',fit)
        summary[name]=dict(**metrics(after),fit_exact=sum(r['exact'] for r in fit),fit_tested=len(fit),candidate_sha256=sha(candidate),training_seconds=log[-1]['seconds'],opening_unchanged=sum(x['turns'][0]['text']==y['turns'][0]['text'] for x,y in zip(before,after)))
        write('results.json',summary);print(json.dumps({name:summary[name]}),flush=True)
if __name__=='__main__':main()
