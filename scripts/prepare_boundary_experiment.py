"""Freeze the $10 Oregon diagnostic and four-arm boundary experiment package."""
import argparse
import hashlib
import io
import tarfile
from pathlib import Path
from boundary_common import contract, write, EXPERIMENT
from canada_narrative import ROOT, digest, read_json
from prepare_canada_pilot import prepare_pilot
from prepare_canada_samples import checkpoint_files


def package(delivery, checkpoints, output):
    config=contract();verified=checkpoint_files(checkpoints)
    manifest=prepare_pilot(delivery,output,region=config['cloud']['region'])
    source=output/'source.tar.gz';previous=digest(source);old=output/'pilot-source.tar.gz';source.rename(old)
    extra={name:ROOT/name for name in read_json(EXPERIMENT/'source.lock.json')}
    extra['experiments/small-model-boundaries-v1/source.lock.json']=EXPERIMENT/'source.lock.json'
    for name in ['BC-B/best.pt','BC-B/result.json']:
        extra['anchor/'+name.split('/',1)[1]]=verified[name]
    hashes={}
    with tarfile.open(source,'w:gz') as dst,tarfile.open(old) as src:
        for entry in src.getmembers():
            if not entry.isfile() or entry.name=='SHA256SUMS':continue
            stream=src.extractfile(entry);h=hashlib.sha256()
            while block:=stream.read(1024*1024):h.update(block)
            hashes[entry.name]=h.hexdigest()
            dst.addfile(entry,src.extractfile(entry))
        for name,path in sorted(extra.items()):
            if name in hashes:raise ValueError('Duplicate package path: '+name)
            dst.add(path,arcname=name);hashes[name]=digest(path)
        checks=''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())).encode()
        info=tarfile.TarInfo('SHA256SUMS');info.size=len(checks);dst.addfile(info,io.BytesIO(checks))
    old.unlink()
    cloud=config['cloud'];minutes=cloud['watchdog_seconds']//60
    script=(output/'user-data.template.sh').read_text().replace(previous,digest(source))
    script=script.replace('shutdown -h +85',f'shutdown -h +{minutes}')
    script=script.replace('timeout 4500 "$PYTHON" scripts/aws_canada_pilot.py --delivery delivery --data prepared --output output',
        f'timeout {cloud["worker_seconds"]} "$PYTHON" scripts/aws_boundary_workload.py --delivery delivery --data prepared --anchor anchor --output output')
    if 'scripts/aws_boundary_workload.py' not in script:raise ValueError('Worker template changed')
    (output/'user-data.template.sh').write_text(script)
    stack=read_json(output/'stack.json')
    code=stack['Resources']['Watchdog']['Properties']['Code']['ZipFile']
    stack['Resources']['Watchdog']['Properties']['Code']['ZipFile']=code.replace('>= 5100',f'>= {cloud["watchdog_seconds"]}')
    write(output/'stack.json',stack)
    compute=manifest['hourly_instance_usd']*(minutes+1)/60
    # Leave a substantial envelope for storage, requests and watchdog overhead.
    if compute+2>cloud['budget_usd']:raise ValueError('Cloud envelope exceeds budget')
    manifest.update(scope='Development decoding diagnostics and four 5M boundary arms; seed 7',
        requested_budget_usd=cloud['budget_usd'],result_file='boundary-result.json',
        workload_timeout_seconds=cloud['worker_seconds'],watchdog_maximum_age_seconds=cloud['watchdog_seconds'],
        controller_wait_seconds=cloud['watchdog_seconds']+300,
        planning_instance_usd_at_watchdog_plus_one_minute=compute,
        planning_other_services_allowance_usd=2,
        shutdown=f'Guest shutdown at exit or +{minutes} minutes; independent AWS watchdog',
        source_files={**manifest['source_files'],**{n:h for n,h in hashes.items() if n.startswith('scripts/') or n.startswith('experiments/')}},
        contract_sha256=digest(EXPERIMENT/'contract.json'),source_lock_sha256=digest(EXPERIMENT/'source.lock.json'),
        anchor_sha256=digest(checkpoints/'BC-B/best.pt'))
    manifest['files']={name:digest(output/name) for name in manifest['files']}
    write(output/'manifest.json',manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['delivery','checkpoints','output']:parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();package(a.delivery,a.checkpoints,a.output)
