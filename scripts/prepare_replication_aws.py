"""Prepare the content-addressed AWS package and a four-hour replication request."""
import argparse
import base64
import gzip
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prefix-data", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = {str(path.relative_to(ROOT)): path for path in (ROOT / "scripts").glob("*.py")}
    files.update({str(path.relative_to(ROOT)): path for path in (ROOT / "tests").glob("test*subword*.py")})
    files["tests/test_replication.py"] = ROOT / "tests/test_replication.py"
    for directory, source in [("data",args.data),("prefix-data",args.prefix_data),("extra",args.extra),("diagnostics",args.diagnostics)]:
        files.update({f"{directory}/{path.name}":path for path in source.iterdir() if path.is_file()})
    for name in ["5m-256","5m-1024"]:
        files[f"models/{name}.pt"] = ROOT / f"models/subword/{name}.pt"
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    checksum = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode()
    package = args.output / "source.tar.gz"
    with package.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in sorted(files):
                info = archive.gettarinfo(str(files[name]), arcname=name)
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                with files[name].open("rb") as source:
                    archive.addfile(info, source)
            info = tarfile.TarInfo("SHA256SUMS")
            info.size = len(checksum)
            archive.addfile(info, io.BytesIO(checksum))
    sha = hashlib.sha256(package.read_bytes()).hexdigest()
    deadline = int(time.time()) + 14400
    prefix = f"experiments/subword-replication-v1/cuda-{sha[:12]}-{deadline}"
    key = f"experiments/subword-replication-v1/packages/{sha}.tar.gz"
    script = (ROOT / "scripts/aws_replication_user_data.sh").read_text()
    for name, value in {"PREFIX": prefix, "SOURCE_KEY": key, "SOURCE_SHA256": sha, "DEADLINE": str(deadline)}.items():
        script = script.replace(f"__{name}__", value)
    request = {"ImageId": "ami-0eb7d782cce2fe526", "InstanceType": "g5.xlarge", "MinCount": 1, "MaxCount": 1,
        "IamInstanceProfile": {"Name": "zero-training-ec2"},
        "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": "subnet-eb3e2f8e", "Groups": ["sg-0059d0413ff74df6e"],
                               "AssociatePublicIpAddress": True, "DeleteOnTermination": True}],
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 75, "VolumeType": "gp3",
                                  "Encrypted": True, "DeleteOnTermination": True}}],
        "InstanceInitiatedShutdownBehavior": "terminate", "ClientToken": f"zero-subword-{sha[:24]}-{deadline}",
        "UserData": base64.b64encode(script.encode()).decode(),
        "TagSpecifications": [{"ResourceType": resource, "Tags": [{"Key": "Project", "Value": "zero"},
              {"Key": "Name", "Value": "subword-replication"}, {"Key": "SourceSha256", "Value": sha},
              {"Key": "Deadline", "Value": str(deadline)}]} for resource in ["instance", "volume"]]}
    request_text = json.dumps(request, indent=2) + "\n"
    (args.output / "request.json").write_text(request_text)
    (args.output / "user-data.sh").write_text(script)
    manifest = {"package_sha256": sha, "package_bytes": package.stat().st_size, "files": hashes,
                "bucket": "zero-training-022118847419", "source_key": key, "result_prefix": prefix,
                "deadline": deadline, "maximum_instance_seconds": 14400,
                "instance_type": "g5.xlarge", "hourly_ec2_usd": 1.006,
                "four_hour_ec2_usd": 4.024, "root_volume_gib": 75, "requested_total_budget_usd": 5,
                "user_data_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "request_sha256": hashlib.sha256(request_text.encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
