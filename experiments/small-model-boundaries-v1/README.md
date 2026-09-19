# Improve the 5M model: boundaries and development diagnostics

This is the first implementation from the small-model research plan. It keeps
5,049,600 parameters and the accepted 2,048-token tokenizer. It tests whether
record order and attention across unrelated stories affect prose quality.
The existing Canada experiment and its frozen implementation remain intact.

The source audit found internal record boundaries in 43.91% of B's original
1,024-token windows and 44.08% of C's. These are observations about packing;
the experiment measures their effect on a newly matched baseline.

## Four arms

| Arm | Record order | Attention |
| --- | --- | --- |
| P0 | Fixed | Across records in a window |
| P1 | Seeded shuffle per pass | Across records in a window |
| P2 | Fixed | Separate records, reset positions |
| P3 | Same shuffle as P1 | Separate records, reset positions |

All arms start with the same seed-7 weights. Each scores exactly 100,499,909
within-record targets from B with the original optimizer and learning-rate
formula. The target roster is selected before shuffling, including the partial
final pass. Its canonical digest must match across arms. Order digests retain
the actual presentation sequence.

A record visit contains inputs `t0 … tn` and labels `t1 … tn, ignore`. Thus
all arms exclude an artificial transition into the next record, and retain the
last token as context for the shared-attention control. P0 is a new matched
baseline. The historical BC-B checkpoint is the development reference.

Packing fills 1,024-token windows across visits. A long visit continues into
the next window; history truncates at the window edge for every arm. Isolated
attention uses visit identity and a causal mask. Positions reset at a visit
change or the beginning of a window. Shared attention uses the original
forward path. Padding and visit-end positions have ignored labels.

Updates group eight windows, with four per microbatch. Loss is divided by the
number of scored targets in the update. The visit-end distribution can change
that count per update; total scored targets, input tokens, windows and planned
updates remain matched. The report includes padding and masked-token counts.

## Diagnostics and samples

The development anchor is the completed BC-B checkpoint. Its identity is
verified when preparing the package and before worker execution.

The worker uses 40 already-opened historical prompts plus 40 authored scene
openings. The authored set contains eight scenario families with five name
variants each. These are development fixtures. Historical windows are grouped
conservatively as one source family; names and repeated templates are not
independent evidence of generalization.

Four anchor decoders are compared: the registered setting, penalty removed,
lower temperature with penalty removed, and greedy. Each produces 256 tokens.
Repetition is reported at 64, 128 and 256 tokens. The worker also records prompt
loss, token frequencies and name tokenization. Prompt losses are descriptive
case/family diagnostics, separate from the registered literary loss metric.

Causal checks cover future-token isolation, cached/full-prefix predictions,
single-record parity, and previous-record interventions. The ordinary
shared-attention model must respond to a changed prefix; the isolated model
must retain the same next-record predictions. Tests also check useful
same-record influence, backward parity and learning a small repeated sequence.

Each trained arm uses the registered decoder on the same 80 prompts. The job
therefore produces 640 continuations in total, plus three blinded 80-pair pages
comparing P0 with each alternative. Human reading supplies coherence and
factual judgments; exact phrase repetition remains its own metric.

## Run and recover

Use the pinned Python dependencies from `canada-narrative-v1/requirements.txt`.
Prepare accepted streams with the existing Canada importer. Then run an arm:

```sh
python scripts/train_boundaries.py --data /path/to/prepared \
  --arm P2 --device cuda --output /path/to/P2
```

Reusing the same output path resumes its last atomic checkpoint. Model,
optimizer, random states, progress, best weights and accounting are restored.
The source/contract/data/roster/device identity must match. Completed results
verify the selected checkpoint hash before reuse. Checkpoints occur at each
200-update selection point and the final update. Failures preserve the last
completed checkpoint.

Generate diagnostics independently:

```sh
python scripts/diagnose_boundaries.py --data /path/to/prepared \
  --checkpoint /path/to/BC-B/best.pt --output /path/to/diagnostics \
  --extra-cases experiments/small-model-boundaries-v1/development-scenes.json
```

Add `--decoder registered` for a single decoder. Each generated case has an
atomic file, a payload hash and the full run identity. Re-running the same
command reuses completed cases; changed inputs or corrupted receipts fail.
Input-only extra prompt fields are enforced. A completed generation leaves
`summary.json`, `mechanism-check.json` and `token-audit.json`.

Freeze the cloud package:

```sh
python scripts/prepare_boundary_experiment.py \
  --delivery /path/to/accepted-delivery \
  --checkpoints /path/to/completed-pilot/results \
  --output /path/to/boundary-package
```

The package includes accepted inputs, the selected anchor, pinned source, and
checksums for every archive member. It uses the existing `launch.py` launch
and collect commands with the package manifest digest. The selected venue is
one Oregon `g5.xlarge`, with a $10 total budget envelope, four-hour worker,
250-minute guest/AWS shutdown guards, and collection/stack cleanup.
Before training, the worker times both attention paths on the same GPU and
checks that the projected training plus evaluation margin fits the envelope.
The price and available image/permissions still require a fresh provider
preflight at launch. `--preflight-only` on the workload performs the timing
and mechanism checks without the four training runs.

The full worker command is:

```sh
python scripts/aws_boundary_workload.py --delivery /path/to/delivery \
  --data /path/to/prepared --anchor /path/to/BC-B --output /path/to/results
```

Cloud bootstrap records dependencies, GPU identity, output logs and the exit
code. Final `boundary-result.json` reports every arm and the order, isolation
and interaction effects in development bits per byte. A negative effect means
lower loss. Runtime and all source/target identities remain in each arm's
result. Resumable generation limits the work repeated after interruption.

## Decision and follow-up

This package is an exploratory development pilot. The accepted historical
validation is shared for checkpoint selection and development comparison using
its existing disjoint selection/outcome windows. Fresh natural-continuation
and factual-family evaluation follow candidate selection. Human decisions and
replication remain separate from successful execution.

Further experiments in simpler-scene curriculum, optimization and textual
state have their own controls and budgets. References: [Braid #34](https://github.com/cenetex/braid/issues/34),
[Crownless #809](https://github.com/atimics/crownlesscarriage/issues/809), and
[early mechanism checks](https://github.com/atimics/zero-grounded-literary-lm/issues/199).

## Checks

```sh
python -m unittest discover -s tests -p test_boundaries.py
```

These tests cover exact target exposure, partial-pass shuffling, attention and
gradient parity, record/causal isolation, same-record influence, training
resume with dropout, changed identities, corrupted sample receipts, tiny
fixture learning, and complete cloud archive/watchdog bindings. Existing
Torch/C and cached-inference checks continue to run in CI.
