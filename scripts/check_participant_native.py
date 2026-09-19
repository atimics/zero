"""Build a checkpoint-specific native probe and compare saved Python samples."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--crownless', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True, help='Matching built native Crownless libraries')
    args = parser.parse_args()
    root, run, build = args.crownless.resolve(), args.run.resolve(), args.build.resolve()
    stage = run / 'native'
    stage.mkdir(exist_ok=False)
    module_path = root / 'tools/language/compile_model.py'
    spec = importlib.util.spec_from_file_location('native_tables', module_path)
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    for subdir in ('assets/language', 'tools/language', 'story'):
        (stage / subdir).mkdir(parents=True)
    shutil.copyfile(run / 'last.ccv2', stage / 'assets/language/core.ccv2')
    shutil.copyfile(root / 'assets/language/tokenizer.json', stage / 'assets/language/tokenizer.json')
    shutil.copyfile(root / 'tools/language/unicode_classes.json', stage / 'tools/language/unicode_classes.json')
    tables.ROOT = stage
    tables.MODEL_SHA = sha(stage / 'assets/language/core.ccv2')
    tables.TOKENIZER_SHA = sha(stage / 'assets/language/tokenizer.json')
    (stage / 'story/cc_core_model_tables.inc').write_text(tables.compile_tables())
    compiler = shutil.which('cc')
    if compiler is None:
        raise ValueError('C compiler unavailable')
    command = [compiler, '-std=c11', '-O2', '-I', str(stage), '-I', str(root / 'src'),
               str(root / 'tools/core_model_probe.c'), str(root / 'src/story/cc_core_model.c'),
               str(build / 'libcrownless_story.a'), str(build / 'libcrownless_sim.a'),
               '-lm', '-o', str(stage / 'probe')]
    receipt = {'command': command, 'model_sha256': tables.MODEL_SHA,
               'tables_sha256': sha(stage / 'story/cc_core_model_tables.inc'),
               'sources': {str(p): sha(p) for p in (module_path, root / 'tools/core_model_probe.c',
                   root / 'src/story/cc_core_model.c', build / 'libcrownless_story.a', build / 'libcrownless_sim.a')}}
    result = subprocess.run(command, capture_output=True)
    (stage / 'build.stdout').write_bytes(result.stdout)
    (stage / 'build.stderr').write_bytes(result.stderr)
    receipt['build_returncode'] = result.returncode
    rows = []
    if result.returncode == 0:
        for sample in json.loads((run / 'samples.json').read_text()):
            actual = subprocess.run([str(stage / 'probe'), str(run / 'last.ccv2'),
                '--participant-prefix', sample['prefix'], '--generate'], capture_output=True, timeout=60)
            # Probe adds exactly one newline even for an unfinished draft.
            output = actual.stdout[:-1] if actual.stdout.endswith(b'\n') else actual.stdout
            expected = bytes.fromhex(sample['bytes_hex'])
            rows.append({'source_line': sample['source_line'], 'returncode': actual.returncode,
                         'stdout_hex': actual.stdout.hex(), 'stderr_hex': actual.stderr.hex(),
                         'bytes_match': output == expected,
                         'completion_match': (actual.returncode == 0) == sample['native_complete']})
    receipt['samples'] = rows
    receipt['passed'] = result.returncode == 0 and bool(rows) and all(
        r['bytes_match'] and r['completion_match'] for r in rows)
    (run / 'native-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
