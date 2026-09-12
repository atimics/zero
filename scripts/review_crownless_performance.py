"""Publish matched model outputs and real turn handoff for listening review."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from crownless_performance import CREATURES, EMOTIONS, VERSION
from crownless_v2 import encode_row, generate
from crownless_v2_export import load_export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=Path('models/crownless-core-v2/tokenizer.json'))
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh review directory')
    manifest = json.loads((args.pilot / 'manifest.json').read_text())
    summary = json.loads((args.pilot / 'results.json').read_text())
    model_path = args.pilot / 'core.ccv2'
    if manifest['schema'] != VERSION: p.error('Unknown performance contract')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != summary['model_sha256']:
        p.error('Model differs from scored artifact')
    torch.set_num_threads(4)
    model, _ = load_export(model_path, args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(x) for x in (args.pilot / 'test.jsonl').read_text().splitlines()]
    scored = {r['id']:r for r in json.loads((args.pilot / 'test-results.json').read_text())['rows']}
    # Pick by input order and task, before inspecting quality.
    base = next(r for r in rows if r['act'] in ('start','question'))
    source_id = base['id'].rsplit(':', 3)[0]
    samples = [scored[r['id']] | {'account': r['prefix']} for r in rows
               if r['id'].rsplit(':', 3)[0] == source_id]
    exchanges = []
    for creature in CREATURES:
        history = [{'speaker':1, 'text':'What happened?'}]
        turns = []
        for turn in range(4):
            speaker = turn % 2
            row = dict(base, performance={'creature':creature if speaker == 0 else 'human',
                                          'emotion':EMOTIONS[turn % 3] if speaker == 0 else 'calm'})
            row['history'] = [{'speaker':'self' if h['speaker'] == speaker else 'other', 'text':h['text']}
                              for h in history[-4:]]
            result = generate(model, tokenizer, encode_row(tokenizer, row, slots=True, conversation=True))
            turns.append({'speaker':speaker, 'performance':row['performance'],
                          'history':row['history'], 'text':result['text'], 'stopped':result['stopped']})
            history.append({'speaker':speaker, 'text':result['text']})
            if not result['stopped']: break
        exchanges.append({'creature':creature, 'account':base['prefix'], 'turns':turns})
    report = {'schema':VERSION, 'model_sha256':summary['model_sha256'],
              'samples':samples, 'exchanges':exchanges,
              'review_status':'Listening and open-ended style judgement await review.'}
    args.output.mkdir(parents=True)
    (args.output / 'review.json').write_text(json.dumps(report, indent=2)+'\n')
    esc = html.escape
    cards = ''
    for i, row in enumerate(samples):
        p = row['performance']
        cards += f'<article><h2>{esc(p["creature"].title())} · {esc(p["emotion"])}</h2><p>{esc(row["text"])}</p><audio controls preload="none" src="sample-{i}.wav"></audio><small>Meaning: {row["meaning"]}; style: {row["identity"]}; emotion: {row["emotion"]}</small></article>'
    chats = ''
    for exchange in exchanges:
        chats += '<section><h2>'+esc(exchange['creature'].title())+' conversation</h2>'
        for turn in exchange['turns']:
            chats += '<p><b>'+esc(turn['performance']['creature'])+':</b> '+esc(turn['text'])+'</p>'
        chats += '</section>'
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Crownless voice study</title><style>body{font:18px/1.5 Georgia,serif;background:#f4eddd;color:#29251e;max-width:1120px;margin:40px auto;padding:20px}h1{font-size:42px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}article,section{background:#fffaf0;padding:22px;border-radius:12px;margin-bottom:18px}h2{font-size:22px}small{display:block;font:13px system-ui}audio{width:100%;margin:12px 0}</style><h1>One account. Nine performances.</h1><p>Human, goblin, and pony speech across calm, fear, and relief. These are the saved model’s words. Voice recordings use fixed synthetic references. Listen for character, feeling, and clear facts.</p><p>'+esc(base['prefix'][2:])+'</p><main>'+cards+'</main>'+chats
    (args.output / 'index.html').write_text(page)
    print(json.dumps({'samples':len(samples), 'exchanges':len(exchanges)}))


if __name__ == '__main__': main()
