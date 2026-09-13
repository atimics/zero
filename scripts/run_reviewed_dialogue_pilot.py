"""Small fixed-budget continuation of the shipped model from approved dialogue."""
import argparse, copy, hashlib, json, random, re, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_v2 import encode_row, batch, generate
from crownless_v2_export import load_export, export

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def normalized(s):return re.sub(r'\d+','N',s.casefold())
APPROVED=['A001','A005','A009','A013','A017','A021','A022','A026','A033','A034','A038','A048']

def make_row(source, output, metadata, history=(), own=True):
    packet=source['packet']
    fields=copy.deepcopy(packet['fields']) if own else []
    copies=[]; used=set()
    for f in fields:
        if not f.get('spoken') or f.get('knowledge')==3:continue
        for m in re.finditer(r'(?<!\w)'+re.escape(f['text'])+r'(?!\w)',output):
            positions=set(range(m.start(),m.end()))
            if used & positions:continue
            used |= positions
            copies.append({**f,'start':len(output[:m.start()].encode()),'end':len(output[:m.end()].encode())})
    known=source['source']['input']
    return dict(id=source['id'],prefix=packet['text'] if own else '', fields=fields,copies=sorted(copies,key=lambda x:x['start']),output=output,kind_id=metadata['meaning_ids'][packet['rule']] if own else 0,confidence=known['confidence'] if own else 80,retold=known['retellings']>=4 if own else False,history=list(history))
def encode(tok,r):return encode_row(tok,r,slots=True,conversation=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--eval',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();assert a.steps>0 and not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(912);rng=random.Random(912)
    model,meta=load_export(a.base,a.tokenizer);model.mode='conversation'
    assert sha(a.base)=='244a809aba71679efe129495bb981ca8d7fbe0d9012948a5f88731dabcb9c48b'
    tok=Tokenizer.from_file(str(a.tokenizer));sources={r['id']:r for r in read(a.input/'accounts.jsonl')}
    drafts={r['source_id']:r for r in json.loads((a.input/'drafts.json').read_text())}
    training=[];replay=[]
    for sid in APPROVED:
        s=sources[sid];assert s['packet']
        for t,line in enumerate(drafts[sid]['lines']):
            history=[dict(speaker='self' if j%2==t%2 else 'other',text=l) for j,l in enumerate(drafts[sid]['lines'][:t])][-4:]
            training.append(make_row(s,line,meta,history,own=t%2==0))
        replay.append(make_row(s,s['model_text'],meta))
    blocked={normalized(sources[i]['source']['input']['account']) for i in APPROVED}
    candidates=[];seen=set()
    for s in read(a.eval):
        if not s['packet']:continue
        key=normalized(s['source']['input']['account'])
        if key in blocked or key in seen:continue
        seen.add(key);candidates.append(s)
    # Include different families before filling further cases, in deterministic order.
    selected=[];kinds=set()
    for s in candidates:
        if s['event']['kind'] not in kinds:selected.append(s);kinds.add(s['event']['kind'])
        if len(selected)==12:break
    assert len(selected)==12
    trainworlds={sources[i]['source']['provenance']['world_seed'] for i in APPROVED}
    assert all(s['source']['provenance']['world_seed'] not in trainworlds for s in selected)
    for name,rows in [('train',training),('replay',replay),('evaluation-accounts',selected)]:
        (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    info=dict(seed=912,steps=a.steps,batch_size=8,learning_rate=5e-5,device='cpu',parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,base_sha256=sha(a.base),tokenizer_sha256=sha(a.tokenizer),approved_sources=APPROVED,train_turns=len(training),replay_rows=len(replay),evaluation_accounts=len(selected),train_worlds=sorted(trainworlds),evaluation_worlds=sorted({s['source']['provenance']['world_seed'] for s in selected}),split_check='Separate seeds and no equal account text after case folding and numeral normalization. Event grammar and names can overlap.',checkpoint_selection='Fixed final step; no test-driven selection.',input_hashes={str(f):sha(f) for f in [a.input/'drafts.json',a.input/'accounts.jsonl',a.eval]},scope='One local pilot, human-approved training targets. Listener gets history only; holder gets its own account. No game integration.')
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    def evaluate(label, both_hold=False):
        outputs=[]
        for s in selected:
            history=[];turns=[]
            for t in range(4):
                hist=[dict(speaker='self' if j%2==t%2 else 'other',text=l) for j,l in enumerate(history)][-4:]
                r=make_row(s,'',meta,hist,own=both_hold or t%2==0)
                result=generate(model,tok,encode(tok,r),max_tokens=80)
                turns.append(dict(text=result['text'],stopped=result['stopped']));history.append(result['text'])
            outputs.append(dict(source_id=s['id'],account=s['source']['input']['account'],confidence=s['source']['input']['confidence'],turns=turns))
        (a.output/(label+'.json')).write_text(json.dumps(outputs,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(evaluation=label,conversations=len(outputs))),flush=True)
        return outputs
    before=evaluate('before')
    before_shared=evaluate('before-shared',True)
    tr=[encode(tok,r) for r in training];rp=[encode(tok,r) for r in replay]
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01);start=time.monotonic();log=[]
    for step in range(1,a.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        inputs=batch(rng.choices(tr,k=6)+rng.choices(rp,k=2),'cpu')
        loss=model.loss(inputs);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==1 or step%25==0:
            item=dict(step=step,loss=loss.item(),seconds=time.monotonic()-start);log.append(item);print(json.dumps(item),flush=True);(a.output/'history.json').write_text(json.dumps(log,indent=2)+'\n')
    export(model,a.tokenizer,a.output/'candidate.ccv2',meta)
    model,meta=load_export(a.output/'candidate.ccv2',a.tokenizer);model.mode='conversation'
    after=evaluate('after')
    after_shared=evaluate('after-shared',True)
    fitted=[]
    for r in training:
        result=generate(model,tok,encode(tok,r),max_tokens=80)
        fitted.append(dict(id=r['id'],history=r['history'],reference=r['output'],text=result['text'],stopped=result['stopped'],exact=result['text']==r['output']))
    (a.output/'training-fit.json').write_text(json.dumps(fitted,indent=2,ensure_ascii=False)+'\n')
    def metrics(rows):
        lines=[t['text'] for r in rows for t in r['turns']]
        return dict(lines=len(lines),distinct_lines=len(set(lines)),stopped=sum(t['stopped'] for r in rows for t in r['turns']),empty=sum(not l for l in lines),same_final_line=max(__import__('collections').Counter(r['turns'][-1]['text'] for r in rows).values()))
    result=dict(before=metrics(before),after=metrics(after),before_shared=metrics(before_shared),after_shared=metrics(after_shared),training_fit_exact=sum(r['exact'] for r in fitted),training_fit_total=len(fitted),candidate_sha256=sha(a.output/'candidate.ccv2'),training_seconds=log[-1]['seconds'])
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
