"""Save matched CPU samples and a C-style repetition control."""
import argparse
import hashlib
import json
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
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tok = Tokenizer.from_file(str(a.tokenizer))
    for name in ['5m-256', '5m-1024', '50m']:
        path = a.large_checkpoint if name == '50m' else a.models / (name+'.pt')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = create(state['config'])
        model.load_state_dict(state['model'])
        result = {'backend': 'CPU FP32', 'seed': 7, 'temperature': .7, 'top_k': 40,
                  'repetition_penalty': 1.0, 'new_tokens': 128, 'step': state.get('step'),
                  'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'tokenizer_sha256': hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(), 'samples': []}
        for prompt in PROMPTS:
            text, ids = sample(model, tok, prompt, return_tokens=True)
            result['samples'].append({'prompt': prompt, 'output': text, 'token_ids': ids,
                                      'prompt_tokens': len(tok.encode(prompt).ids)})
        (a.output / ('samples-'+name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        if name == '50m':
            prompt = PROMPTS[4]
            base = {**result['samples'][4], 'repetition_penalty': 1.0}
            text, ids = sample(model, tok, prompt, return_tokens=True, repetition_penalty=1.1)
            control = {**base, 'output': text, 'token_ids': ids, 'repetition_penalty': 1.1}
            record = {k: v for k, v in result.items() if k not in ['samples', 'repetition_penalty']}
            record.update(recent_tokens=64, method='Subtract log(penalty) from each distinct recent token logit before temperature and top-k, matching C probability division.', samples=[base, control])
            (a.output / 'repetition-control.json').write_text(json.dumps(record, indent=2)+'\n')
        print(name, flush=True)

if __name__ == '__main__':
    main()
