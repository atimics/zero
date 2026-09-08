# Gutenberg 5M on AWS

This branch adds a CUDA backend for the same 4,852,992-parameter ZERO model.
It imports the native C initialization and exports native checkpoints for the
existing int8 exporter. RMS normalization, rotary positions, tied embeddings,
GELU, residual dropout, clipping, AdamW groups, and learning-rate schedule match
the C architecture and formulas. Sampling and dropout use PyTorch's RNG; the
GPU training trajectory is a separate run with seed 7.

The local parity test compares C and PyTorch probabilities, all gradients, one
optimizer update, and checkpoint round trips. The full model's initial loss on
one 512-character window differs by less than one millionth on CPU. The AWS
bootstrap also checks that loss on the actual CUDA device before training.

The prepared request uses one g5.xlarge in us-east-1, with an A10G 24 GB GPU.
The live AWS price query returned $1.006 per hour for Linux on-demand compute.
The job has a two-hour wall-clock limit including bootstrap, for about $2.012
of EC2 compute plus small storage and IPv4 charges. The budget target is $3.
The root disk is encrypted and deleted on termination. The existing training
role and egress-only security group are reused.

Training uses the same 100,000 updates, batch 2, 102.4 million character
presentations, seed 7, learning rate 0.0003, 1,000-step warmup, and cosine decay.
FP32 is used with TF32 disabled. At update 500, the job requires at least 20,000
characters per second and enough time to finish within the remaining limit.
The job saves checkpoints before this check. It uploads progress every two
minutes, exports the best model on completion, uploads results, and shuts down.
Instance-initiated shutdown terminates the instance.

`last-rng.pt` retains the Torch RNG states alongside native weights and AdamW
moments. The native checkpoint header retains the original C RNG field; use an
explicit Torch continuation plan to resume the GPU random sequence.

## Prepare and check

```sh
uv venv --python 3.12 /tmp/zero-torch
uv pip install --python /tmp/zero-torch/bin/python torch==2.8.0 numpy==2.2.6
/tmp/zero-torch/bin/python -m unittest discover -s tests -p test_zero_torch.py
cc -O2 -std=c11 tests/torch_reference.c -o /tmp/zero-initialize -lm
/tmp/zero-initialize initialize /tmp/zero-initial.ckpt
/tmp/zero-torch/bin/python scripts/prepare_zero_aws.py \
  --data /private/tmp/zero-gutenberg-data/ready \
  --initial /tmp/zero-initial.ckpt --output /tmp/zero-aws-package
aws ec2 run-instances --region us-east-1 \
  --cli-input-json file:///tmp/zero-aws-package/request.json --dry-run
```

The package contains code, a native initial checkpoint, the three corpus splits,
and file hashes. The request contains a deadline, so regenerate it immediately
before launch. Upload the package under the content-addressed S3 key in
`manifest.json`, then use that exact request for the authorized launch.

CUDA speed remains a live calibration result. The local C run can continue
while this path is prepared and checked.
