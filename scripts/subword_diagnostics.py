"""Freeze token placement statistics and a source-name mixing proxy."""
import argparse,collections,json,re
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

STOP=set('The This That These Those A An And Or But If When While Then There Here Where What Who Which Why How I He She It We They His Her Their Our My Your You Its As At In On Of To For From With Without By Into Upon After Before One Two Three Mr Mrs Miss Sir Lady Lord Dr Saint Chapter Book Part Volume January February March April May June July August September October November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday Yes No Now Well Oh O God English England London French France America American'.split())

def index_names(path):
    counts=collections.defaultdict(collections.Counter);titles={}
    for line in path.open():
        row=json.loads(line);book=str(row['metadata']['book_id']);titles[book]=row['metadata']['title'];text=row['text']
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b',text):
            word=m.group();before=text[:m.start()].rstrip()
            if word in STOP or not before or before[-1] in '.!?\n':continue
            counts[word][book]+=1
    return {'method':'Capitalized words inside sentences; at least five occurrences and exactly one training book. Common words excluded. This is a source-name proxy, with ambiguous names and false positives.','names':{w:dict(c) for w,c in counts.items() if len(c)==1 and sum(c.values())>=5},'books':titles,'stop_words':sorted(STOP)}

def placement_index(data):
    tok=Tokenizer.from_file(str(data/'tokenizer.json'));text=[tok.decode([i]) for i in range(tok.get_vocab_size())]
    ids=np.fromfile(data/'train.bin',dtype='<u2');starts=np.array([bool(t and t[0].isspace()) for t in text]);ends=np.array([bool(t and t[-1].isspace()) for t in text]);alpha=np.array([bool(t.lstrip() and t.lstrip()[0].isalpha()) for t in text]);previous=np.concatenate(([True],ends[ids[:-1]]));word_start=(previous|starts[ids])&alpha[ids]
    counts=np.bincount(ids[word_start],minlength=len(text));return {'counts':counts.tolist(),'token_text':text}

def analyze(samples,index,names,tok):
    rows=[]
    for row in samples:
        if 'token_ids' in row:ids=row['token_ids'];start=row['prompt_tokens'];source='original generated token IDs'
        else:ids=tok.encode(row['output']).ids;start=len(tok.encode(row['prompt']).ids);source='retokenized decoded output; original generation IDs unavailable'
        unusual=[];opportunities=0
        for j in range(max(1,start),len(ids)):
            text=index['token_text'][ids[j]];previous=index['token_text'][ids[j-1]]
            if text.lstrip() and text.lstrip()[0].isalpha() and ((text and text[0].isspace()) or (previous and previous[-1].isspace())):
                opportunities+=1
                if index['counts'][ids[j]]==0:unusual.append({'position':j,'token':ids[j],'text':text})
        body=tok.decode(ids[start:]) if 'token_ids' in row else row['output'][len(row['prompt']):];words=set(re.findall(r'\b[A-Z][a-z]{2,}\b',body));matched={w:list(names['names'][w]) for w in sorted(words) if w in names['names']};books=sorted({b for ids in matched.values() for b in ids})
        rows.append({'prompt':row['prompt'],'seed':row.get('seed',7),'model':row.get('model'),'token_source':source,'word_initial_opportunities':opportunities,'unusual_initial_count':len(unusual),'unusual_initial_examples':unusual[:8],'matched_source_names':matched,'distinct_source_books':len(books),'source_books':books,'capitalized_words':len(words),'matched_name_coverage':len(matched)/max(1,len(words))})
    opportunities=sum(r['word_initial_opportunities'] for r in rows);unusual=sum(r['unusual_initial_count'] for r in rows)
    return {'samples':len(rows),'unusual_initial_count':unusual,'word_initial_opportunities':opportunities,'unusual_initial_rate':unusual/max(1,opportunities),'passages_with_multiple_source_books':sum(r['distinct_source_books']>1 for r in rows),'rows':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--training-jsonl',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    names=index_names(a.training_jsonl);index=placement_index(a.data)
    (a.output/'source-names.json').write_text(json.dumps(names,indent=2)+'\n');(a.output/'token-placement.json').write_text(json.dumps(index,indent=2)+'\n')
    raw=json.loads(a.samples.read_text());rows=raw['samples'] if isinstance(raw,dict) else raw;result=analyze(rows,index,names,Tokenizer.from_file(str(a.data/'tokenizer.json')));(a.output/'baseline-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
