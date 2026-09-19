# Participant training pilot

This pipeline trains one person's next speech or action from a private simulator
view and heard speech. Crownless PRs #813 and #817 supply the view, compact
compiler and native entry point. Teacher samples come from two independent
sessions per pair. The student consumes one actor's target with input loss masked.

## Local smoke results

Four CPU runs used the same six dragon-scene turns, seed 19 and batch size 2.
The architecture has 4,945,153 parameters. Fresh and warm arms consumed identical
target-token budgets at each step count. Warm initialization uses the shipped
row-scaled int8 checkpoint loaded back into float weights.

| Initialization | Steps | Target tokens | Last batch loss | Two sampled replies |
| --- | ---: | ---: | ---: | --- |
| Fresh | 10 | 2,549 | 7.3851 | repeated spaces; both reach the limit |
| Warm | 10 | 2,549 | 1.0084 | malformed mixed phrases; both reach the limit |
| Fresh | 100 | 25,278 | 3.3450 | malformed JSON; one emits EOS |
| Warm | 100 | 25,278 | 0.0159 | one exact training reply; one mixed reply reaches the limit |

The exported int8 models were compiled into separate native probes using
checkpoint-specific tables. Python and C match all eight output byte sequences
and completion results. This proves the tested export and inference path.
The samples also show why output quality and valid participant JSON need their
own checks. A completed byte sequence can still be malformed JSON.

These are small pipeline tests on training examples. All six input rows are
pending review and were admitted through the explicit `--smoke` option. Broader
quality, generalization, personal memory and action behavior require reviewed
data and held-out encounters. The 100-step warm sample shows that the model can
learn the new turn format on a familiar example; it establishes a working path
for the rebuild.

## Reproduction

Use Python with `scripts/requirements-crownless-v2.txt`. Build the participant
version of Crownless, including `core_model_probe` and the static libraries.
Compile paired candidate rows with Crownless's `participant_training.py`.

```sh
python scripts/train_crownless_participant.py \
  --train /tmp/compact/previews.jsonl \
  --reference /path/to/crownless/assets/language/core.ccv2 \
  --tokenizer /path/to/crownless/assets/language/tokenizer.json \
  --output /tmp/participant-warm-100 --initialization warm \
  --smoke --steps 100 --batch-size 2 --seed 19 --device cpu
python scripts/check_participant_native.py \
  --run /tmp/participant-warm-100 --crownless /path/to/crownless \
  --build /path/to/crownless-build
```

Use `--initialization fresh` for the matched fresh run. Use fresh output paths
for each run. The native checker builds its generated tables in the run folder;
its source and library hashes identify the code it used.

For a training comparison, omit `--smoke`, supply `--validation`, and mark rows
with `source_review_status: approved`, `review_status: approved_compact` and a
`world_group`. A world group must include forks and related histories of the
same simulated world. The loader checks actor-only shifted labels against the
tokenizer, rejects excluded targets, and checks world overlap and duplicated
long targets across splits. Semantic review must check factual support after
context selection. The final checkpoint is used; validation loss is weighted
by target count. Model selection and a separate test set belong in the larger
experiment plan.

## Evidence and tests

`smoke-results.json.gz` maps artifact names to their exact text. It preserves
all four manifests, histories, samples, native receipts, the six compiled rows,
training-source versions and dependency versions. Early 10-step and later
100-step trainer source snapshots have hashes matching their manifests.
SHA-256: `c9098f5174e22894a93cab6586850a7f226db7db8d3d8e54c5b161af09eb5f87`.
Checkpoint files remain in the local run directories named in the receipts.

Eight participant tests pass, covering real gradient updates, target weighting,
review status, actor boundaries, corrupted labels and tokens, split leakage and
failed native setup receipts. The existing `test_crownless_v2.py` reports one
local failure in `test_published_model_speaks_with_new_names`: the older model
produces "In Newhaven in Éva put up a notice about Flood relief." The existing
model, model code and that test are unchanged in this branch. Its failure is
preserved in the evidence archive and remains separate from participant parity.
