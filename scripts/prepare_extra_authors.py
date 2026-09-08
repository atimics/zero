"""Freeze two new author test streams using existing cleaning and tokenizer."""
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from build_gutenberg_corpus import clean_book

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--training-text',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(a.data/'tokenizer.json'));training=' '.join(a.training_text.read_text().split());manifest={}
    for author,book in [('shelley',84),('stoker',345)]:
        url=f'https://gutenberg.pglaf.org/cache/epub/{book}/pg{book}.txt';raw=urllib.request.urlopen(url,timeout=60).read();text=clean_book(raw);(a.output/f'{author}.txt').write_text(text)
        ids=[i for line in text.splitlines(keepends=True) for i in tok.encode(line).ids];np.asarray(ids,dtype='<u2').tofile(a.output/f'{author}.bin')
        starts=np.linspace(1024,len(ids)-64,1024,dtype=np.int64);keep=[];excluded=[]
        for start in starts:
            span=tok.decode(ids[int(start)-128:int(start)+64]);words=span.split()[-64:]
            if len(words)==64 and ' '.join(words) in training:excluded.append(int(start))
            else:keep.append(int(start))
        (a.output/f'{author}-starts.json').write_text(json.dumps(keep))
        manifest[author]={'book_id':book,'source_url':url,'raw_sha256':hashlib.sha256(raw).hexdigest(),'clean_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':len(ids),'candidate_windows':1024,'retained_windows':len(keep),'excluded_64_word_overlap_starts':excluded,'overlap_method':'Exact whitespace-normalized final 64 words in the prefix-plus-target span, searched against the complete training text.'}
        print(author,manifest[author],flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
