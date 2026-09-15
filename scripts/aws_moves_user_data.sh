#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/crownless-moves.log) 2>&1

# Filled by the reviewed launch request.
BUCKET=zero-training-022118847419
PREFIX=__PREFIX__
SOURCE_KEY=__SOURCE_KEY__
SOURCE_SHA256=__SOURCE_SHA256__
DEADLINE=__DEADLINE__
EXPERIMENT=__EXPERIMENT__
COMMIT=__COMMIT__
export AWS_DEFAULT_REGION=us-east-1
export CROWNLESS_SOURCE_COMMIT="$COMMIT"
mkdir -p /opt/crownless-run
cd /opt/crownless-run

# Absolute deadline includes boot and dependency installation.
remaining=$((DEADLINE - $(date +%s)))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
shutdown -h +$(((remaining + 59) / 60))

# A rejected run is a result, not a failure: the trainer exits non-zero when no
# checkpoint clears the gate, and its candidate weights are the point of the
# exercise. Everything under output/ ships either way.
finish() {
  exit_code=$?
  trap - EXIT
  printf '{"exit_code":%s,"train_exit":%s,"experiment":"%s","finished_at":%s}\n' \
    "$exit_code" "${train_exit:-null}" "$EXPERIMENT" "$(date +%s)" > finish.json
  if command -v aws >/dev/null; then
    timeout 120 aws s3 cp /var/log/crownless-moves.log "s3://$BUCKET/$PREFIX/bootstrap.log" || true
    timeout 60 aws s3 cp finish.json "s3://$BUCKET/$PREFIX/finish.json" || true
    if [ -d output ]; then timeout 600 aws s3 sync output "s3://$BUCKET/$PREFIX/output/" || true; fi
  fi
  shutdown -h now
}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv awscli
aws s3 cp "s3://$BUCKET/$SOURCE_KEY" source.tar.gz
echo "$SOURCE_SHA256  source.tar.gz" | sha256sum -c -
tar -xzf source.tar.gz
sha256sum -c SHA256SUMS
python3 -m venv /opt/crownless-venv
/opt/crownless-venv/bin/pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/opt/crownless-venv/bin/pip install -q numpy==2.2.6 tokenizers==0.22.0
/opt/crownless-venv/bin/pip freeze > environment.txt
aws s3 cp environment.txt "s3://$BUCKET/$PREFIX/environment.txt"
/opt/crownless-venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

(
  while sleep 180; do
    if [ -d output ]; then
      aws s3 sync output "s3://$BUCKET/$PREFIX/progress/" --exclude '*.tmp' --only-show-errors || true
    fi
  done
) &
sync_pid=$!

set +e
/opt/crownless-venv/bin/python scripts/train_crownless_moves.py \
  --corpus corpus --output output --base base/core.ccv2 --tokenizer base/tokenizer.json \
  --chat chat-rows-stance.jsonl --device cuda 2>&1 | tee train.log
train_exit=${PIPESTATUS[0]}
set -e
kill "$sync_pid" 2>/dev/null || true

mkdir -p output
cp train.log output/train.log
printf '{"train_exit":%s,"experiment":"%s","commit":"%s","deadline":%s}\n' \
  "$train_exit" "$EXPERIMENT" "$COMMIT" "$DEADLINE" > output/run.json
find output -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > output/SHA256SUMS
