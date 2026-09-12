# Crownless 5M v2: shared accounts and learned speech

12 September 2026. This package trains a **4,935,937-parameter core** that writes
one sentence from a selected Crownless account meaning. The saved 8-bit model
passes all **1,250 frozen paired test rows**, with all required names intact. It
also passes the 1,250 source-paraphrase rows when supplied their meaning labels.
The game-run check passes **119 of 121 distinct account cases**; the remaining
two cases need another source pattern in the native parser.

The model and [shared game grammar](https://github.com/atimics/crownlesscarriage/pull/689)
form one pipeline. The game chooses an NPC's held account. Its parser supplies
the meaning, field roles, permitted details, and uncertainty. The small model
writes the sentence and selects whole fields from that account. This is a
measured first core for controlled speech. Broader English and voice training
can build on this task boundary.

## What the experiments changed

The initial text decoder confused names and meanings. Adding field labels and
a copy head helped. Replacing names with field markers inside the language
state improved copying further. Some events still mixed two different meanings:
completed masonry and repairs delayed by a stone shortage shared an event kind.

The selected `packet` mode supplies the specific recognized meaning. Its
encoder sorts spoken fields by field ID. Names remain in a literal source
buffer. A copy decision emits those bytes and feeds the corresponding marker
back into the decoder. Source wording and older event lines then have a stable
representation for this one-account task.

These are the completed 48-pattern development comparisons, each with 2,000
updates, seed 17, and 96 rows per split:

| Input and output mechanism | Approved meaning, usual wording | Approved meaning, new source wording | Required names, usual wording |
|---|---:|---:|---:|
| Plain text | 4/96 | 0/96 | 0/88 |
| Field labels and event kind | 2/96 | 0/96 | 0/88 |
| Field labels and literal copying | 37/96 | 21/96 | 57/88 |
| Field markers and literal copying | 92/96 | 50/96 | 88/88 |

Those four arms share the backbone, tokenizer, data, update schedule, and total
parameter allocation. The plain-text arm leaves its extra feature and copy
parameters inactive. The later 48-meaning packet pilot passed 192/192 rows in
each split. That pilot also changed spoken-field selection and therefore
measures the whole revised recipe.

The final grammar adds older harvest, trade, raid, cult, and dragon accounts.
It has **61 patterns across 44 of the 136 event kinds** in Crownless schema 95.
A kind can require several source patterns. Coverage is recorded per kind and
the native parser checks each supported source shape.

The approved-form score checks complete sentences against both legal variants
and the permitted uncertainty endings. It catches role reversal and wrong
certainty even when every name appears. Other output forms require review.
The score measures this grammar's meanings and expression forms. The supplied
labels make the source-paraphrase score a check of representation consistency;
learning arbitrary English parsing remains a separate task. The deterministic
reference renderer already expresses these authored rules exactly. The learned
core's present value is a tested foundation for further expression training.

## Final training and saved model

The corpus was frozen with generator seed **20260919** after development. It
contains 25,000 training rows, 1,250 validation rows, 1,250 test rows, and 1,250
test paraphrases. Training has 459,416 target BPE tokens and 2,035,030 output
bytes. Whole names are separate across splits. Paired examples change one
field or check an invariant output. All rule families occur in training; the
held-out axes are field values, names, combinations, and source wording.

The source records identify these as authored schema counterfactuals. Actual
game accounts are evaluated separately. Files retain UTF-8 field offsets,
knowledge labels, copy labels, grammar identity, and split hashes.

The model uses eight causal attention layers, width 192, six heads, SwiGLU
width 624, RMSNorm, rotary positions, tied 4,096-entry byte-BPE embeddings,
field features, 62 meaning entries including the default, and a span-copy
head. Its maximum context is 512 tokens. Training starts from fresh weights
with output-only supervision. This diagnostic uses Crownless examples alone.

Each final run uses 3,000 updates, batch 16, AdamW, a 0.0004 peak learning rate,
100 warm-up updates, linear decay, and gradient clipping at 1. Validation
selects the checkpoint within each run. Seed **17** was selected for the saved
artifact before final scoring; its selected checkpoint is update **1,500**.
Seeds **29** and **43** check repeatability on 244 rows per split, covering each
pattern twice as paired cases. Results and source identities are in the
adjacent `results.json` and `evidence` directory.

| Final run | Approved meanings per split | Required names per split |
|---|---:|---:|
| Seed 17, full frozen set | 1,250/1,250 | 1,170/1,170 |
| Seed 29, repeat check | 244/244 | 228/228 |
| Seed 43, repeat check | 244/244 | 228/228 |
| Saved 8-bit seed 17 | 1,250/1,250 | 1,170/1,170 |

Both the usual-source and source-paraphrase splits have these scores. The
8-bit export gives exactly the same text and stopping behavior as float
weights on all 2,500 frozen inputs and all 119 recognized game cases.

`models/crownless-core-v2/core.ccv2` contains **5,044,760 bytes** including scales
and metadata. The export stores matrices at 8 bits and vectors at float32.
The Python reference loader expands those weights to float32. The artifact
test runs complete generated sentences from the saved bytes. Tests also check
cache parity, causal attention, UTF-8 copying, changed names, role order,
meaning scoring, and damaged payload rejection.

## Game-run check

Seed 17 over 120 simulation days supplied 10,448 spoken observations across
19 event kinds. After merging duplicates with the same held account and
confidence band, 121 cases remained. The native parser recognized 119 and
the exported model gave an approved full meaning in all 119. Two cases of
bandit recruitment use a further source pattern; both remain explicit misses
in `evidence/game-q8.json`.

This world helped select the added grammar patterns during development. It is
an integration check on real held accounts. Fresh-world generalization needs
a further reserved set of simulation seeds.

Examples generated by the saved model:

> Mara Venn posted a notice about Relief charter in Thornford.

> Thornford's drought harvest fell short of what the eastern settlements needed.

> Silverwick was short of its food reserve target.

The shared grammar also improves ordinary game gossip: the same world produced
3,630 spoken rows before the change and 10,448 after it. Unsupported observed
accounts fell from 3,409 to zero in that window. These observations include
wording variants and account changes. The native game renderer has broader
fallback coverage than this first learned packet model.

On an M4 Max CPU with four Torch threads, warm generation for the recognized
game accounts took about **18 ms median / 27 ms p95**. This measures sentence
generation in the Python float32 reference after compact-weight loading.
Startup, parsing, tokenization, peak memory, native neural inference, and WASM
performance need their own measurements. Other local work was active during
the runs, so these are practical timings rather than isolated speed benchmarks.

## Reproduce

Build the account tools and corpus using the instructions in Crownless's
`docs/core-v2-accounts.md`. The corpus source commit is `a60e1430`; the manifest
records the full commit and hashes. From this repository:

```sh
python -m pip install -r scripts/requirements-crownless-v2.txt
python scripts/test_crownless_v2.py
python scripts/train_crownless_v2.py --data ../crownless/out/core-v2-data --output out/core-v2-seed17 --steps 3000 --batch-size 16 --modes packet --seed 17 --eval-rows 1250 --device mps
python scripts/score_crownless_v2.py --data ../crownless/out/core-v2-data --results out/core-v2-seed17/packet
python scripts/crownless_v2_export.py --checkpoint out/core-v2-seed17/packet/best.pt --tokenizer out/core-v2-seed17/tokenizer.json --output out/core-v2-seed17/core.ccv2
python scripts/evaluate_crownless_v2.py --model models/crownless-core-v2/core.ccv2 --tokenizer models/crownless-core-v2/tokenizer.json --data ../crownless/out/core-v2-data --output out/core-v2-q8-evaluation
```

Use `--device cpu` or `--device cuda` on those systems. For the repeatability
runs, use seeds 29 and 43, a fresh output folder, `--eval-rows 244`, and
`--tokenizer out/core-v2-seed17/tokenizer.json`.

Collect the game world with `crownless_gossip_corpus --seed 17 --days 120` and
save stdout as JSONL. `scripts/evaluate_crownless_game.py --help` lists the
paths needed to score those held accounts through the native parser.

The next useful data expansion is reviewed variation in sentence structure
and partial knowledge. Each new meaning rule should improve native gossip,
carry a fresh test family, and preserve existing account fidelity.
