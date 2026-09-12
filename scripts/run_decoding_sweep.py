"""Run the fixed 12-setting, six-prompt, eight-seed decoding grid."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from subword_model import create, sample
from train_subword import PROMPTS
from measure_sample_failures import repetition


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=True)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = create(state['config']).eval()
    model.load_state_dict(state['model'])
    tok = Tokenizer.from_file(str(a.tokenizer))
    meta = {'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'tokenizer_sha256': hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
            'backend': 'CPU FP32, two threads, KV cache', 'step': state['step'], 'new_tokens': 128,
            'temperatures': [.6, .7, .8], 'top_k': [20, 40], 'repetition_penalties': [1.0, 1.1], 'seeds': list(range(1, 9)), 'prompts': PROMPTS,
            'claim': 'Exploratory decoding comparison. Report repetition and diversity by prompt. Blind human coherence ratings remain separate.'}
    (a.output / 'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    with (a.output / 'samples.jsonl').open('w') as f:
        for temperature, top_k, penalty in itertools.product(meta['temperatures'], meta['top_k'], meta['repetition_penalties']):
            start = time.perf_counter()
            for prompt in PROMPTS:
                for seed in meta['seeds']:
                    output, ids = sample(model, tok, prompt, seed=seed, return_tokens=True, temperature=temperature, top_k=top_k, repetition_penalty=penalty, use_cache=True)
                    body = output[len(prompt):]
                    metrics = repetition(body)
                    row = dict(temperature=temperature, top_k=top_k, repetition_penalty=penalty, prompt=prompt, seed=seed, output=output, token_ids=ids, prompt_tokens=len(tok.encode(prompt).ids), repetition=metrics)
                    f.write(json.dumps(row)+'\n'); f.flush()
            print(temperature, top_k, penalty, 'seconds', round(time.perf_counter()-start, 2), flush=True)

if __name__ == '__main__':
    main()
