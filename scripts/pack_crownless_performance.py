"""Keep reproducible training evidence, a checked model, and compact listening audio."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import wave


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('first','pilot','fresh','review','references','output','model-output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.model_output.exists(): p.error('Use fresh package directories')
    result = json.loads((args.pilot/'results.json').read_text())
    manifest = json.loads((args.pilot/'manifest.json').read_text())
    if result['model_sha256'] != sha(args.pilot/'core.ccv2'): p.error('Scored model differs')
    review = json.loads((args.review/'review.json').read_text())
    fresh = json.loads((args.fresh/'results.json').read_text())
    if review['model_sha256'] != result['model_sha256'] or fresh['model_sha256'] != result['model_sha256']:
        p.error('Review or final evaluation uses another model')
    args.output.mkdir(parents=True); args.model_output.mkdir(parents=True)
    shutil.copy2(args.pilot/'core.ccv2', args.model_output/'core.ccv2')
    (args.model_output/'performance.json').write_text(json.dumps({
        'schema':manifest['schema'], 'model_sha256':result['model_sha256'],
        'tokenizer_sha256':manifest['tokenizer_sha256'], 'grammar_sha256':manifest['grammar_sha256'],
        'parameters':manifest['parameters'], 'selected_step':result['selected_step'],
        'runtime':'Python conversation encoder with explicit creature and emotion prefix'}, indent=2)+'\n')
    for label, source in (('first',args.first),('followup',args.pilot),('fresh',args.fresh)):
        destination = args.output/label; destination.mkdir()
        for path in sorted(source.iterdir()):
            if path.suffix not in ('.json','.jsonl'): continue
            if path.name in ('manifest.json','results.json','history.json') and path.stat().st_size < 50000:
                shutil.copy2(path,destination/path.name)
            else:
                (destination/(path.name+'.gz')).write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    destination = args.output/'listening'; destination.mkdir()
    for name in ('review.json','audio.json','index.html'):
        shutil.copy2(args.review/name,destination/name)
    references = destination/'references'; references.mkdir()
    import lameenc
    receipts = []
    paths = [(path,destination) for path in sorted(args.review.glob('sample-*.wav'))]
    paths += [(path,references) for path in sorted(args.references.glob('*.wav'))]
    for source, target in paths:
        with wave.open(str(source),'rb') as wav:
            if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getnframes()==0:
                raise ValueError('Expected nonempty mono PCM16 audio')
            encoder = lameenc.Encoder(); encoder.set_bit_rate(48)
            encoder.set_in_sample_rate(wav.getframerate()); encoder.set_channels(1); encoder.set_quality(2)
            audio = encoder.encode(wav.readframes(wav.getnframes())) + encoder.flush()
        output = target/source.with_suffix('.mp3').name; output.write_bytes(audio)
        receipts.append({'file':str(output.relative_to(destination)), 'sha256':sha(output), 'source_wav_sha256':sha(source)})
        if source.with_suffix('.json').exists(): shutil.copy2(source.with_suffix('.json'), target/source.with_suffix('.json').name)
    page = (destination/'index.html').read_text().replace('.wav"','.mp3"')
    (destination/'index.html').write_text(page)
    (destination/'compressed_audio.json').write_text(json.dumps({'codec':'MP3', 'bitrate_kbps':48, 'files':receipts},indent=2)+'\n')
    inventory = {str(path.relative_to(args.output)):sha(path) for path in sorted(args.output.rglob('*')) if path.is_file()}
    (args.output/'SHA256.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({'files':len(inventory), 'bytes':sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
                      'model_bytes':(args.model_output/'core.ccv2').stat().st_size}))


if __name__ == '__main__': main()
