#!/usr/bin/env bash
# Train the generalist on one AWS spot GPU, bring the results home, and leave nothing behind.
#
#   experiments/generalist/train_aws.sh                 (AWS_PROFILE and AWS_REGION as usual)
#   STEPS=30000 MAX_MINUTES=120 INSTANCE_TYPES="g5.xlarge g6.xlarge" experiments/generalist/train_aws.sh
#
# Same shape as crownless2's tools/train_aws.sh:
#   1. a throwaway key pair and a security group that admits SSH from this machine's public IP only;
#   2. one spot GPU instance on the AWS Deep Learning AMI (PyTorch preinstalled); it powers itself off after
#      MAX_MINUTES whatever happens, and power-off terminates it;
#   3. copies train_cuda.py, scripts/zero_torch.py, initial.ckpt and data/, then trains;
#   4. copies output/ back to runs/<name>/;
#   5. terminates the instance and deletes the key pair and security group, even if a step failed.
set -euo pipefail

REGION=${AWS_REGION:-$(aws configure get region || echo us-east-1)}
INSTANCE_TYPES=${INSTANCE_TYPES:-"g5.xlarge g6.xlarge g5.2xlarge g4dn.xlarge"}
MAX_MINUTES=${MAX_MINUTES:-150}
STEPS=${STEPS:-30000}
NAME=zero-generalist-$(date +%Y%m%d%H%M%S)
HERE=$(cd "$(dirname "$0")" && pwd)
ZERO=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d)
KEY="$WORK/key.pem"
AWS="aws --region $REGION"
INSTANCE="" SG="" KEYNAME=""

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup() {
    status=$?
    log "cleaning up"
    if [ -n "$INSTANCE" ]; then
        $AWS ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null 2>&1 || true
        $AWS ec2 wait instance-terminated --instance-ids "$INSTANCE" 2>/dev/null || true
        log "instance $INSTANCE terminated"
    fi
    if [ -n "$SG" ]; then
        for i in 1 2 3 4 5 6; do $AWS ec2 delete-security-group --group-id "$SG" 2>/dev/null && break; sleep 10; done
        log "security group $SG deleted"
    fi
    if [ -n "$KEYNAME" ]; then $AWS ec2 delete-key-pair --key-name "$KEYNAME" >/dev/null 2>&1 || true; log "key pair $KEYNAME deleted"; fi
    rm -rf "$WORK"
    exit $status
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

for f in train_cuda.py initial.ckpt data/mining.valid.utility.jsonl data/dragon.valid.txt; do
    [ -e "$HERE/$f" ] || { echo "missing $HERE/$f (run build_data.py and make the initial checkpoint first)"; exit 1; }
done
$AWS sts get-caller-identity >/dev/null || { echo "not logged in to AWS (try: aws sso login)"; exit 1; }

MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '\n')
VPC=$($AWS ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" != "None" ] || { echo "no default VPC in $REGION"; exit 1; }
AMI=$($AWS ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id --query Parameter.Value --output text)

KEYNAME=$NAME
$AWS ec2 create-key-pair --key-name "$KEYNAME" --query KeyMaterial --output text > "$KEY"; chmod 600 "$KEY"
SG=$($AWS ec2 create-security-group --group-name "$NAME" --description "zero generalist training (temporary)" --vpc-id "$VPC" --query GroupId --output text)
$AWS ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
log "key pair and security group ready (SSH from $MYIP only)"

USERDATA=$(printf '#!/bin/bash\nshutdown -h +%d\n' "$MAX_MINUTES" | base64 | tr -d '\n')
for TYPE in $INSTANCE_TYPES; do
    log "asking for a spot $TYPE"
    if INSTANCE=$($AWS ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEYNAME" --security-group-ids "$SG" \
            --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
            --instance-initiated-shutdown-behavior terminate --user-data "$USERDATA" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":75,"VolumeType":"gp3","DeleteOnTermination":true,"Encrypted":true}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" --query 'Instances[0].InstanceId' --output text 2>"$WORK/err"); then
        break
    fi
    INSTANCE=""; log "  not available: $(tail -1 "$WORK/err")"
done
[ -n "$INSTANCE" ] || { echo "no spot capacity for any of: $INSTANCE_TYPES"; exit 1; }
log "instance $INSTANCE ($TYPE) starting"
$AWS ec2 wait instance-running --instance-ids "$INSTANCE"
HOST=$($AWS ec2 describe-instances --instance-ids "$INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 ubuntu@$HOST"
for i in $(seq 1 90); do $SSH true 2>/dev/null && break; sleep 5; done
log "connected to $HOST"

tar -c -C "$ZERO" scripts/zero_torch.py experiments/generalist/train_cuda.py experiments/generalist/initial.ckpt experiments/generalist/data \
    | $SSH 'mkdir -p zero && tar -x -C zero'
PY='source /opt/pytorch/bin/activate 2>/dev/null || true; python3'
DEADLINE=$(( $(date +%s) + MAX_MINUTES * 60 - 300 ))
log "training $STEPS steps on $($SSH 'nvidia-smi --query-gpu=name --format=csv,noheader')"
RUN="$HERE/runs/$NAME"
mkdir -p "$RUN"
START=$(date +%s)
$SSH "cd zero/experiments/generalist && $PY train_cuda.py --data data --initial initial.ckpt --output output --steps $STEPS --deadline $DEADLINE" 2>&1 | tee "$RUN/train.log"
log "trained in $(( $(date +%s) - START ))s"
$SSH "cd zero/experiments/generalist && tar -c output" | tar -x -C "$RUN"
log "results in $RUN/output"
