"""Measure shared wording in preserved samples; exclude the supplied prompt."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def analyze(data):
    rows=[]
    for a,b in zip(data['models']['5m-256']['samples'],data['models']['5m-1024']['samples']):
        assert a['prompt']==b['prompt']
        prompt=a['prompt'];texts=[r['output'][len(prompt):] for r in [a,b]]
        words=[re.findall(r"\w+|[^\w\s]",t) for t in texts]
        overlap={}
        for n in [3,4]:
            sets=[set(tuple(w[i:i+n]) for i in range(len(w)-n+1)) for w in words]
            common=sets[0]&sets[1];union=sets[0]|sets[1]
            overlap[str(n)]={'shared':len(common),'jaccard':len(common)/max(1,len(union)),'examples':[' '.join(x) for x in sorted(common)[:12]]}
        rows.append({'prompt':prompt,'overlap':overlap})
    return {'method':'Case-sensitive words and punctuation; prompt excluded. Shared n-grams describe overlap, not training-data memorization.','rows':rows}
if __name__=='__main__':
    path=ROOT/'docs/subword/results.json';out=ROOT/'docs/subword/overlap.json';out.write_text(json.dumps(analyze(json.loads(path.read_text())),indent=2)+'\n')
