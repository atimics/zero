"""Publish the thirteen-act corpus to a Hugging Face dataset repository.

Verifies every split against the manifest hashes before uploading, so a corpus
that was edited or truncated after its build cannot be published by accident.
Needs HF_TOKEN with write access in the environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--repo', required=True, help='Target dataset repo, e.g. someone/crownless-core-acts')
    p.add_argument('--private', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Verify and list, upload nothing')
    p.add_argument('--token-parameter', help='SSM SecureString holding the token, e.g. /crownless/HF_TOKEN')
    args = p.parse_args()

    manifest = json.loads((args.corpus / 'manifest.json').read_text())
    for split, entry in manifest['splits'].items():
        actual = sha(args.corpus / f'{split}.jsonl')
        if actual != entry['sha256']:
            p.error(f'{split}.jsonl does not match the manifest: {actual} != {entry["sha256"]}')
    files = sorted(x for x in args.corpus.iterdir() if x.is_file())
    total = sum(x.stat().st_size for x in files)
    print(json.dumps({'repo': args.repo, 'verified_splits': list(manifest['splits']),
                      'files': [x.name for x in files], 'bytes': total,
                      'acts': manifest['acts']['total']}, indent=2))
    if args.dry_run:
        print('dry run: nothing uploaded')
        return
    from huggingface_hub import HfApi, get_token
    # SSM first: a SecureString keeps the token out of shell history, process
    # listings, and any transcript, and the same parameter serves a cloud
    # worker later. Falls back to the environment or a stored login.
    token = None
    if args.token_parameter:
        import subprocess
        token = subprocess.check_output(
            ['aws', 'ssm', 'get-parameter', '--name', args.token_parameter,
             '--with-decryption', '--query', 'Parameter.Value', '--output', 'text'],
            text=True).strip()
    token = token or os.environ.get('HF_TOKEN') or get_token()
    if not token:
        p.error('No token: pass --token-parameter, set HF_TOKEN, or run "hf auth login"')
    api = HfApi(token=token)
    print(json.dumps({'authenticated_as': api.whoami()['name']}))
    api.create_repo(args.repo, repo_type='dataset', private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(args.corpus), repo_id=args.repo, repo_type='dataset',
                      commit_message=f"Crownless core acts v1 ({manifest['acts']['total']} acts, "
                                     f"{manifest['splits']['train']['rows']} train rows)")
    print(json.dumps({'published': f'https://huggingface.co/datasets/{args.repo}'}))


if __name__ == '__main__': main()
