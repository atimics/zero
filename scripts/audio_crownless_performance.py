"""Design synthetic creature references, or voice the model's matched outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

REFERENCE = 'The road is quiet this morning. Come inside, and tell me where you are going. We can find a place for your horses.'
DESCRIPTIONS = {
    'human': 'A fictional adult woman with a warm clear middle voice and a light Welsh accent. Calm conversational English, steady pace, clear consonants. Clean studio speech.',
    'goblin': 'A fictional small adult goblin with a wiry raspy nasal voice, a low mischievous chuckle in the tone, quick clipped consonants, and clear conversational English. A distinctive fantasy creature, alert and curious. Calm delivery. Clean studio speech.',
    'pony': 'A fictional magical talking pony with a bright rounded musical voice, soft warm resonance, bouncy gentle rhythm, and very clear conversational English. Friendly and curious with an airy playful quality. Calm delivery. Clean studio speech.',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('design','speak'), required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--review', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--device', choices=('cpu','mps','cuda'), default='cpu')
    args = p.parse_args()
    os.environ['HF_HOME'] = str(args.cache)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    from scipy.io.wavfile import write
    torch.set_num_threads(2)

    def wav(path, samples, rate):
        samples = np.asarray(samples).reshape(-1)
        if not np.isfinite(samples).all() or not .15 <= len(samples)/rate <= 40:
            raise ValueError('Invalid generated speech')
        peak = float(np.abs(samples).max())
        if peak < .001: raise ValueError('Generated speech is silent')
        write(path, rate, (samples * min(1, .82/peak) * 32767).astype(np.int16))

    if args.mode == 'design':
        if args.references.exists(): p.error('Use a fresh reference directory')
        from qwen_tts import Qwen3TTSModel
        from huggingface_hub import snapshot_download
        model_id = 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
        snapshot = snapshot_download(model_id, local_files_only=True)
        model = Qwen3TTSModel.from_pretrained(snapshot, device_map=args.device,
                    dtype=torch.float32, attn_implementation='eager')
        args.references.mkdir(parents=True)
        for i, (name, description) in enumerate(DESCRIPTIONS.items()):
            torch.manual_seed(20260912+i)
            started = time.monotonic()
            with torch.inference_mode():
                samples, rate = model.generate_voice_design(text=REFERENCE, language='English',
                    instruct=description, max_new_tokens=500)
            path = args.references / f'{name}.wav'
            wav(path, samples[0], rate)
            record = {'model':model_id, 'description':description, 'text':REFERENCE,
                      'seed':20260912+i, 'sha256':sha(path), 'seconds':time.monotonic()-started,
                      'packages':{n:importlib.metadata.version(n) for n in ('torch','qwen-tts')},
                      'snapshots':[p.name for p in (args.cache/'hub'/'models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign'/'snapshots').iterdir()]}
            path.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps({'reference':name, 'seconds':record['seconds']}), flush=True)
    else:
        if args.review is None: p.error('Supply the review directory')
        from pocket_tts import TTSModel
        report = json.loads((args.review/'review.json').read_text())
        model = TTSModel.load_model(language='english')
        states = {name:model.get_state_for_audio_prompt(str(args.references/f'{name}.wav')) for name in DESCRIPTIONS}
        receipts = []
        for i, sample in enumerate(report['samples']):
            name = sample['performance']['creature']
            torch.manual_seed(20260912)
            started = time.monotonic()
            with torch.no_grad():
                chunks = list(model.generate_audio_stream(states[name], sample['text']))
            samples = torch.cat(chunks).detach().cpu().numpy()
            path = args.review/f'sample-{i}.wav'
            if path.exists(): p.error('Preserve the earlier audio in a separate review directory')
            wav(path, samples, model.sample_rate)
            receipts.append({'file':path.name, 'text':sample['text'], 'performance':sample['performance'],
                'sha256':sha(path), 'reference_sha256':sha(args.references/f'{name}.wav'),
                'generation_seconds':time.monotonic()-started, 'audio_seconds':len(samples)/model.sample_rate,
                'seed':20260912})
            print(json.dumps({'sample':i, 'seconds':receipts[-1]['generation_seconds']}), flush=True)
            (args.review/'audio.json').write_text(json.dumps({'model':'Pocket TTS',
                'packages':{n:importlib.metadata.version(n) for n in ('torch','pocket-tts')},
                'snapshots':[p.name for p in (args.cache/'hub'/'models--kyutai--pocket-tts'/'snapshots').iterdir()],
                'scope':'Fixed reference per creature; emotion reaches Pocket through spoken wording. Acoustic control requires listening review.',
                'samples':receipts}, indent=2)+'\n')


if __name__ == '__main__': main()
