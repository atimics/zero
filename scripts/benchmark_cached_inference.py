"""Check trained-checkpoint logits, sample IDs, and local cache speed."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from subword_model import create, sample
from train_subword import PROMPTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--large-checkpoint', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    tok = Tokenizer.from_file(str(a.tokenizer))
    result = {'backend': 'CPU FP32, two threads', 'torch': torch.__version__, 'platform': platform.platform(), 'models': {}}
    for name in ['5m-256', '5m-1024', '50m']:
        path = a.large_checkpoint if name == '50m' else a.models / (name+'.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = create(state['config']).eval()
        model.load_state_dict(state['model'])
        ids = tok.encode(('The girl opened the door. ' * 250)).ids[:min(128, model.context)]
        maximum = 0.
        with torch.no_grad():
            logits, cache = model.forward_cached(torch.tensor([ids[:64]]))
            reference = model(torch.tensor([ids[:64]]))[:, -1:]
            maximum = max(maximum, (reference-logits).abs().max().item())
            torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)
            for end in range(65, 81):
                logits, cache = model.forward_cached(torch.tensor([[ids[end-1]]]), cache)
                reference = model(torch.tensor([ids[:end]]))[:, -1:]
                maximum = max(maximum, (reference-logits).abs().max().item())
                torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)
        agreements = []
        for prompt in PROMPTS:
            _, plain = sample(model, tok, prompt, return_tokens=True)
            _, cached = sample(model, tok, prompt, return_tokens=True, use_cache=True)
            agreements.append({'prompt': prompt, 'identical_ids': plain == cached})
        timing = []
        for label, prompt, count in [('short', PROMPTS[1], 128), ('near_context_limit', tok.decode(tok.encode('The girl opened the door. '*250).ids[:model.context-4]), 8)]:
            sample(model, tok, prompt, count=2, use_cache=True)
            seconds = {False: [], True: []}
            outputs = {}
            for repeat in range(3):
                for cached in ([False, True] if repeat % 2 == 0 else [True, False]):
                    start = time.perf_counter()
                    output = sample(model, tok, prompt, count=count, return_tokens=True, use_cache=cached)
                    seconds[cached].append(time.perf_counter()-start)
                    outputs[cached] = output[1]
            plain = statistics.median(seconds[False]); cached = statistics.median(seconds[True])
            timing.append({'case': label, 'prompt_tokens': len(tok.encode(prompt).ids), 'new_tokens': count, 'uncached_seconds': seconds[False], 'cached_seconds': seconds[True], 'median_speedup': plain/cached, 'identical_ids': outputs[False] == outputs[True]})
        result['models'][name] = {'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'max_absolute_logit_difference': maximum, 'sample_agreement': agreements, 'timing': timing}
        a.output.write_text(json.dumps(result, indent=2)+'\n')
        print(name, maximum, [(r['case'], r['median_speedup'], r['identical_ids']) for r in timing], flush=True)

if __name__ == '__main__': main()
