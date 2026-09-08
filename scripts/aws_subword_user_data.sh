#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/zero-gutenberg.log) 2>&1

# Filled by the reviewed launch request.
BUCKET=zero-training-022118847419
PREFIX=__PREFIX__
SOURCE_KEY=__SOURCE_KEY__
SOURCE_SHA256=__SOURCE_SHA256__
DEADLINE=__DEADLINE__
export AWS_DEFAULT_REGION=us-east-1
mkdir -p /opt/zero-run
cd /opt/zero-run

# Absolute deadline includes boot and dependency installation.
remaining=$((DEADLINE - $(date +%s)))
if [ "$remaining" -le 0 ]; then shutdown -h now; exit 1; fi
shutdown -h +$(((remaining + 59) / 60))

finish() {
  exit_code=$?
  trap - EXIT
  printf '{"exit_code":%s,"finished_at":%s}\n' "$exit_code" "$(date +%s)" > finish.json
  if command -v aws >/dev/null; then
    timeout 60 aws s3 cp /var/log/zero-gutenberg.log "s3://$BUCKET/$PREFIX/bootstrap.log" || true
    timeout 60 aws s3 cp finish.json "s3://$BUCKET/$PREFIX/finish.json" || true
    if [ -d output ]; then timeout 90 aws s3 sync output "s3://$BUCKET/$PREFIX/output/" || true; fi
  fi
  shutdown -h now
}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv awscli gcc
aws s3 cp "s3://$BUCKET/$SOURCE_KEY" source.tar.gz
echo "$SOURCE_SHA256  source.tar.gz" | sha256sum -c -
tar -xzf source.tar.gz
sha256sum -c SHA256SUMS
python3 -m venv /opt/zero-venv
/opt/zero-venv/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
/opt/zero-venv/bin/pip install numpy==2.2.6 tokenizers==0.22.0
/opt/zero-venv/bin/pip freeze > environment.txt
aws s3 cp environment.txt "s3://$BUCKET/$PREFIX/environment.txt"
/opt/zero-venv/bin/python -m unittest discover -s tests -p test_subword.py
/opt/zero-venv/bin/python scripts/evaluate_character_baseline.py --data data --checkpoint character.ckpt --output output/character-baseline.json
(
  while sleep 120; do
    if [ -d output ]; then
      aws s3 sync output "s3://$BUCKET/$PREFIX/progress/" --exclude '*.tmp' --only-show-errors || true
    fi
  done
) &
sync_pid=$!
/opt/zero-venv/bin/python scripts/train_subword.py --data data --output output --deadline "$DEADLINE"
kill "$sync_pid" || true
find output -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > output/SHA256SUMS
