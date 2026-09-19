# AWS CUDA timing package

Current result: AWS capacity blocked the L4 attempts in all three Canadian
zones, the larger L4 host, and the approved A10G host with automatic zone
selection. All six temporary stacks are deleted. GPU instance time was zero.
See `launch-result.json`, `a10-launch-state.json` and `a10-collection.json`.
The user approved the A10G package in `a10-proposal-manifest.json` under the
US$2 budget. AWS returned `InsufficientInstanceCapacity` on execution.
The approved A10G rate was US$1.117/hour. Timing remains pending capacity.

The user approved an Oregon A10G timing run under the same US$2 ceiling.
Prepare it with `--region us-west-2 --instance-type g5.xlarge --automatic-zone`.
The Oregon profile pins image `ami-0d105fd7469b31d32`, VPC `vpc-80728ce6`,
and US$1.006/hour. Refresh the AWS image, network and price checks before
launch. The source archive and training workload remain identical to the
Canada timing package. The raw workload cost projection uses the original
L4 rate; calculate Oregon cost from measured hours at US$1.006/hour.

On September 19, the approved Oregon A10G timing workload passed. The median
update took 0.059478 seconds; the 95th percentile was 0.059879 seconds.
The CPU/CUDA maximum logit difference was 0.01136 against the 0.05 limit.
The measured runtime was PyTorch 2.8.0+cu129 on NVIDIA A10G. The image already
contained that build; the torch 2.8.0 version requirement accepted it.

The synthetic projection, including the 30% margin, is 8.66 minutes for A/B
and 32.04 minutes for B/C: 40.70 minutes total. At the checked Oregon rate,
training and checkpoint selection cost about US$0.682 in compute. Setup,
checkpoint I/O, final scoring and generation add time. These are planning
estimates from synthetic updates. Corpus training presentations remain zero.
Runtime receipts and raw results stay outside Git. Collection hashes matched,
the instance reached terminated, and the temporary stack reached
DELETE_COMPLETE. The timing run and its cleanup are complete.

Run one `g6.xlarge` in `ca-central-1` with the accepted experiment's model
and training update. The workload uses synthetic token IDs, five warm-up
updates and 100 measured updates. It compares initial CUDA bfloat16 logits
with CPU float32 logits, checks finite training, and reports median and
95th-percentile update time plus estimated A/B and B/C hours and instance cost.

The review budget is **US$2**. The instance price checked for Linux on-demand
is US$0.8936/hour. The 31-minute planning amount is US$0.4617 for EC2.
Storage, public IPv4, S3 requests, Lambda and Scheduler add small charges.
The budget is a spending limit for this operation; AWS billing is separate
from the runtime controls.

The pinned image is `ami-092e8a40239c8733d`, an AWS-owned PyTorch 2.8 Ubuntu
24.04 image. The existing Canada subnet is `subnet-a24506fe`. The package
creates a private temporary S3 bucket, an outbound-HTTPS security group,
scoped instance permissions, a Lambda watchdog, and a one-minute schedule.
The schedule expires 25 hours after preparation. Launch settings expire
after 24 hours. The bucket's objects expire after one day as a cleanup backup.

Three runtime controls apply:

1. The CUDA workload has a 900-second timeout.
2. Guest shutdown is scheduled at boot for 30 minutes, with immediate shutdown
   after success or failure. EC2 shutdown behavior is termination.
3. The independent AWS watchdog terminates tagged instances at age 30 minutes
   or later, including instances whose boot failed. It checks every minute.

AWS scheduling and API delays can extend the nominal limit. The launcher
also watches the run and calls termination after its local wait limit.
It confirms termination, copies results locally, empties the temporary
bucket, and deletes the temporary stack. If the local process stops, use
`collect` to recover the run and complete cleanup. The AWS watchdog continues
independently of that local process.

Prepare a concrete package:

```sh
python3 scripts/prepare_canada_aws.py --output /tmp/zero-canada-aws-timing
```

Review `manifest.json`, `stack.json`, `request.template.json`, and
`user-data.template.sh`. All package files, including the launcher, are
hash-bound. The source archive includes pinned code, contracts and input
hashes. The corpus stays in its existing location.

After approval of this exact manifest hash, execute:

```sh
python3 /tmp/zero-canada-aws-timing/launch.py launch \
  --directory /tmp/zero-canada-aws-timing \
  --approved-manifest-sha256 APPROVED_MANIFEST_SHA256
```

Recover the same run with `collect` in place of `launch`. The client token
and run tag remain fixed. The launcher checks the AWS account, package
hashes, stack ownership and existing launch state.

Results include `timing.json`, the GPU description, installed dependencies,
bootstrap log, exit receipt, collection hashes and cleanup state. The A10G launch has user approval; its first attempt completed cleanup after
AWS reported insufficient capacity.

## Canada cloud pilot

The four-arm pilot uses one g5.xlarge in Canada Central, a US$2 ceiling,
a 75-minute workload timeout, and guest plus independent AWS termination
at 85 minutes. Input rights evidence is scoped to Canada. Corpus data and
checkpoints use the Canada region. Oregon timing supplies a planning estimate.

The worker rebuilds the pinned streams and verifies the accepted preparation
hash before training. It runs AB-A, AB-B, BC-B and BC-C with seed 7, for
255,314,200 scored token presentations. It validates both pairs, checks selected
checkpoint hashes, and evaluates the protected outcome windows after all arms
finish. Generation, repetition checks and blind human review follow separately;
training completion alone leaves the quality decision open.

Prepare from the accepted delivery:

```sh
python scripts/prepare_canada_pilot.py --delivery /path/to/accepted-delivery \
  --output /path/to/pilot-package
```

Launch and recovery use the packaged `launch.py` and manifest hash, as in the
GPU timing workflow. `pilot.json` records completed training and paired loss
scores. The collector saves nested checkpoints and hashes every result file.
Runtime inputs, checkpoints and receipts stay outside Git.
