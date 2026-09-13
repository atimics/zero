# AWS CUDA timing package

Current result: AWS capacity blocked the L4 attempts in all three Canadian
zones, the larger L4 host, and the approved A10G host with automatic zone
selection. All six temporary stacks are deleted. GPU instance time was zero.
See `launch-result.json`, `a10-launch-state.json` and `a10-collection.json`.
The user approved the A10G package in `a10-proposal-manifest.json` under the
US$2 budget. AWS returned `InsufficientInstanceCapacity` on execution.
The approved A10G rate was US$1.117/hour. Timing remains pending capacity.

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
