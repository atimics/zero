# Canada narrative experiment

Goal: improve a small local model's scene continuation, character continuity,
event order, and dialogue. A/B tests the curated collection. B/C tests broader
reading against repeated reading at the same target-token budget.

This package registers the experiment and prepares the accepted corpus. The
training outcome is pending. Corpus training presentations: **zero**.

## Model and data

Use `atimics/zero` and its existing `5m-1024` model: vocabulary 2,048, context
1,024, width 256, six layers, eight heads, feed-forward width 960. The delivery
uses this repository's frozen byte BPE. The separate ZERO.5 lab uses a different
tokenizer. This package follows the delivered control and tokenizer.

`inputs.lock.json` pins the accepted release IDs, source JSONL hashes, control
files, acceptance evidence, and merge receipt. The importer checks these
identities, every training record's content hash, reversible encoding, exact
record token count, and B/C membership. It encodes the ASCII text line by line
with the frozen tokenizer. Training records join in frozen order with their
original bytes. Record offsets remain in separate index files.

| Comparison | Control | Candidate | Scored tokens per arm | Pilot seed |
| --- | --- | --- | ---: | ---: |
| A/B | Preserved A | Accepted B | 27,157,191 | 7 |
| B/C | Accepted B | Accepted C | 100,499,909 | 7 |

Both arms start with identical fresh weights. Each comparison has its own
cosine schedule. B/C starts fresh too. Seeds 17 and 29 are registered for
replication after a passing pilot. The default seed is 7.

The faster runner groups four sequences per microbatch. Each update still
scores 8,192 targets. CPU runs use eight threads. Apple MPS and CUDA runs use
two host threads. Validation groups four windows per forward pass.
`speed-checks.json` records the CPU/GPU sweep and full-model numerical checks.
The Apple GPU was about four times faster than the two-thread baseline in
the same short sweep. Its two-update repeat check had a maximum weight
difference of 6.82e-7. All recorded engineering tolerances passed.

The runner visits every target position in order, then cycles. Each input is
the previous stream token. The first input is the last stream token. Contexts
can cross record boundaries. This treats all arms as continuous streams.
The final partial update pads its context and masks extra targets. Budgets
count scored target presentations, while the extra input positions remain
part of the measured compute. C gets exactly one target pass; B gets three
full passes and 19,028,336 additional targets in B/C.

## Shared evaluation and decisions

The preserved A validation stream is the common evaluation source. The B/C
delivery protects this text from training overlap. The first half supplies
64 fixed checkpoint-selection windows. The second half supplies 256 fixed
outcome windows and 200 fixed writing prompts. Each score window has 256
targets and a 768-token prefix. Checkpoint selection uses the lowest bits
per byte at each 200-update report and at the final update. The earliest
checkpoint wins ties. Both arms finish before the outcome scorer runs.

Each candidate must reduce outcome bits per byte by at least 2%. Its mean
repeated four-gram excess may rise by at most 0.01. A designated reviewer
completes 200 blind pairs and must prefer the candidate in at least 120 cases,
with at least 160 decisive choices. Skip covers ties and unclear choices.
The page uses A/B/skip, five-pair rounds, undo, autosave, and resume.
The scorer reports a conditional Wilson interval on decisive choices.
This is evidence about this reviewer and these prompts.

The review rubric covers stable characters, sensible actions, dialogue,
readability, and looping. These are human judgments. Prediction and repetition
remain separate numerical measures. The delivered annotated scenes are
available for later source-grounded diagnostics. Their annotation text stays
in the delivery evidence. Test content stays sealed.

## Prepare and measure

Use Python 3.11 or 3.12. Run from the repository root:

```sh
python3 -m venv /tmp/zero-canada-env
/tmp/zero-canada-env/bin/pip install -r experiments/canada-narrative-v1/requirements.txt
/tmp/zero-canada-env/bin/python scripts/canada_narrative.py \
  --delivery /absolute/path/to/accepted-delivery \
  --output /tmp/zero-canada-prepared
/tmp/zero-canada-env/bin/python scripts/run_canada_narrative.py benchmark \
  --device mps --output /tmp/zero-canada-timing.json
```

`preparation.json` records the verified stream hashes. The trainer checks the
prepared manifest against this receipt. `implementation.lock.json` binds the
runner, model code, review code, contract, input lock, preparation receipt,
and direct dependency versions. `timing.json` records the Apple GPU synthetic
measurement. `cpu-baseline-timing.json` preserves the earlier two-thread CPU
measurement. The projections include periodic selection scoring and a 30%
planning margin. Checkpoint I/O, final scoring, and generation add time.
Cloud cost requires a selected host and timing on that host.

## Run the four pilot arms

Choose one device for the pair. The examples use the Mac GPU through MPS.
Use `--device cpu` for CPU execution or `--device cuda` on AWS.
A CUDA run requires bfloat16 support. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA operations. The
following commands each run a full registered arm:

```sh
/tmp/zero-canada-env/bin/python scripts/run_canada_narrative.py train \
  --device mps --data /tmp/zero-canada-prepared --comparison AB --arm A --output /tmp/canada-AB-A
/tmp/zero-canada-env/bin/python scripts/run_canada_narrative.py train \
  --device mps --data /tmp/zero-canada-prepared --comparison AB --arm B --output /tmp/canada-AB-B
/tmp/zero-canada-env/bin/python scripts/run_canada_narrative.py train \
  --device mps --data /tmp/zero-canada-prepared --comparison BC --arm B --output /tmp/canada-BC-B
/tmp/zero-canada-env/bin/python scripts/run_canada_narrative.py train \
  --device mps --data /tmp/zero-canada-prepared --comparison BC --arm C --output /tmp/canada-BC-C
```

Each arm writes starting-weight identity, training history, last and best
checkpoints, and a completion receipt with exact presentations. A partial
run keeps its last reported checkpoints. A fresh output directory starts a
fresh run; automatic resume is a later addition.

## Review a completed pair

```sh
/tmp/zero-canada-env/bin/python scripts/review_canada_narrative.py build \
  --data /tmp/zero-canada-prepared --comparison AB \
  --control /tmp/canada-AB-A --candidate /tmp/canada-AB-B --output /tmp/canada-AB-review
/tmp/zero-canada-env/bin/python scripts/serve_ab_review.py \
  --directory /tmp/canada-AB-review --responses /tmp/canada-AB-events.jsonl
```

Use the existing collector with the generated packet and page. Open
the local address printed by the collector for the review. Keep `blind-key.json` separate from the
reviewer's page. Candidate placement is balanced 100/100 across A and B.
After collecting event JSONL, compute the decision:

```sh
/tmp/zero-canada-env/bin/python scripts/review_canada_narrative.py summarize \
  --directory /tmp/canada-AB-review --events /tmp/canada-AB-events.jsonl \
  --reviewer REVIEWER_ID_FROM_EXPORT --output /tmp/canada-AB-decision.json
```

Use the same commands with BC and its two run directories for B/C. Passing
the pilot supplies the evidence for the registered repeat seeds. For each
repeat arm, add `--seed 17 --pilot-decision /tmp/canada-AB-decision.json`
or seed 29 and use a fresh output directory. B/C repeats use their B/C
decision. The runner checks the passing pilot's seed, comparison, contract,
and prepared manifest. Report
each repeat seed separately and require both to pass the numerical gates.

## Checks

```sh
/tmp/zero-canada-env/bin/python -m unittest discover -s tests -p test_canada_narrative.py
```

Focused tests cover exact cyclic targets and final masks, paired weights,
finite backward passes, score boundaries, frozen hashes, line encoding,
review completion, skips, duplicate events, revisions, and invalid pairs.
The existing parity and review checks also run in CI.

Reproduce the full Mac speed sweep and numerical checks with:

```sh
/tmp/zero-canada-env/bin/python scripts/benchmark_canada_narrative.py \
  --output /tmp/canada-speed-checks.json
```

## AWS timing proposal

A read-only AWS query confirmed `g6.xlarge` and `g5.xlarge` offerings in
Canada Central. On-demand Linux pricing checked on 2026-09-12 is
US$0.8936/hour for `g6.xlarge` and US$1.117/hour for `g5.xlarge`.
The [AWS G6 specification](https://aws.amazon.com/ec2/instance-types/g6/)
lists an NVIDIA L4 GPU with 24 GB of GPU memory. Prices come from the
[official Canada catalog](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/ca-central-1/index.json).

Proposed next step: time synthetic updates on one `g6.xlarge` in
`ca-central-1`, then choose the full run using its measured time. A
15-minute instance interval is about US$0.22; setup time and storage add
cost. The runtime command is:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 /tmp/zero-canada-env/bin/python \
  scripts/run_canada_narrative.py benchmark --device cuda --output /tmp/canada-cuda-timing.json
```

The benchmark uses synthetic token IDs. It needs the pinned code and
dependencies. Full corpus runs use the accepted delivery in Canada.
The user approved a US$2 timing run. AWS capacity blocked the L4 and A10G
attempts. All six temporary stacks are deleted, with zero GPU runtime.
See [launch receipts](aws/launch-result.json). AWS timing remains pending capacity.
