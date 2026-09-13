#!/bin/bash
set -Eeuo pipefail
mkdir -p /opt/zero-timing/output
cd /opt/zero-timing
exec > >(tee -a output/bootstrap.log) 2>&1
export AWS_DEFAULT_REGION=ca-central-1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PIP_DISABLE_PIP_VERSION_CHECK=1
BUCKET=__BUCKET__
SOURCE_SHA=__SOURCE_SHA__

# The AWS watchdog also covers failure before this script starts.
shutdown -h +30
finish() {
  code=$?
  trap - EXIT
  printf '{"exit_code":%s,"finished_at":%s}\n' "$code" "$(date +%s)" > output/finish.json
  timeout 90 aws s3 sync output "s3://$BUCKET/results/" --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

timeout 60 aws s3 cp "s3://$BUCKET/source.tar.gz" source.tar.gz --only-show-errors
echo "$SOURCE_SHA  source.tar.gz" | sha256sum -c -
tar -xzf source.tar.gz
sha256sum -c SHA256SUMS
PYTHON=/opt/pytorch/bin/python
if [ ! -x "$PYTHON" ]; then PYTHON=/opt/conda/envs/pytorch/bin/python; fi
if [ ! -x "$PYTHON" ]; then
  python3 -m venv /opt/zero-timing-env
  PYTHON=/opt/zero-timing-env/bin/python
fi
timeout 720 "$PYTHON" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
timeout 180 "$PYTHON" -m pip install numpy==2.2.6 tokenizers==0.22.0
"$PYTHON" -m pip freeze > output/environment.txt
nvidia-smi > output/gpu.txt
timeout 900 "$PYTHON" scripts/aws_canada_workload.py --output output/timing.json
