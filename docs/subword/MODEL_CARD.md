# ZERO Gutenberg Subword 5M

Two experimental English prose completion models: **5m-256** and **5m-1024**.
Each has 5,049,600 parameters and a shared 2,048-token byte-level BPE vocabulary.
They differ in context length: 256 or 1,024 tokens. This release contains both
selected checkpoints, exact recorded samples, validation curves, and a local
comparison demo.

## Results

Lower bits per byte means better prediction of held-out text.

| Model | Validation bits/byte | Test bits/byte | Selected update |
| --- | ---: | ---: | ---: |
| Character baseline, 4.85M, 512 characters | 1.619902 | 1.638699 | 98,500 |
| Subword 5.05M, 256 tokens | 1.552254 | 1.568483 | 11,900 |
| Subword 5.05M, 1,024 tokens | 1.550287 | 1.563721 | 11,800 |

The long-context model's test score is 0.004762 bits/byte lower than the
short-context model, a 0.30% reduction in the original training seed.
The completed five-seed replication finds a mean difference of +0.000060
bits/byte, with 95% interval [-0.005193, +0.005313]. The contexts meet the
registered practical-equivalence rule; the direction is unresolved. Sample pairs support observations about those particular
outputs; a coherence advantage would require a larger, scored comparison.

The subword models improve test bits/byte over the character baseline by 4.28%
and 4.58%. That comparison changes tokenization, text per context, training
exposure, initialization, optimizer details, and numerical precision together.
It measures the combined recipe.

## Evaluation protocol

The tokenizer was trained exclusively on the training split. Books are split
by author. Validation authors are Hawthorne, Melville, and Trollope; the test
author is Alcott. Their works are held out from training.

Checkpoint selection uses 64 fixed validation windows every 100 updates, plus
the last update. Each window scores the same 64 target BPE tokens in both
models. Starts are evenly spaced from token 1,024 through the last valid target
span. The 256-token model receives 192 tokens before its 64 prediction
positions; the 1,024-token model receives 960. Both receive the same target text.

Final evaluation uses 1,024 fixed windows per split, with 65,536 target BPE
tokens: 206,379 bytes in validation and 207,306 bytes in test. Token losses are
summed in natural-log units, then divided by log(2) and the exact number of
target bytes. This is sampled evaluation of the held-out corpus.

The character baseline predicts the exact decoded target bytes, using up to
its 512-character context. Its scores here use this shared-target protocol;
the earlier character-model report used a different set of windows.

The two subword runs share initialization seed 7, initial weights, a separate
seeded training-block generator, token budget, optimizer and learning-rate
schedule. Every update chooses the same contiguous 8,192-token block. The
short model divides it into 32 sequences and the long model into eight. Target
exposure is identical; available context and dropout trajectories differ.
Training difficulty also changes: there are 32 sequence starts per block in
the short arm versus 8 in the long arm. Mean available input context is about
128.5 versus 512.5 tokens. This compares training contexts and inference
contexts together.

Both selected checkpoints occur near the end of a fixed 12,208-update cosine
schedule. Training runs to the full 100M-token budget; early stopping is absent.
The finding is scoped to that budget. Longer training could change the result.

Validation curves record 123 checks for each model. Their raw histories are
in [results.json](results.json), and [curves.svg](curves.svg) shows the full
selection history. The test split is scored after checkpoint selection.
The original scores above use one training seed and one held-out test author.
The completed replication is reported below.

## Replication and conditional window analysis

A CPU FP32 rescore saved all 1,024 window losses for each split. A paired
20,000-draw bootstrap gives long-minus-short test bits/byte -0.004859, with
95% window interval [-0.007108, -0.002611]; the long model wins 560 windows.
Validation gives -0.001960 with interval [-0.004273, 0.000355]. These intervals
are conditional on the trained weights and treat sampled windows as exchangeable.
Within-book correlation, training-seed variance and new-author variance remain
outside their scope. The CUDA BF16 rescore reproduces the original -0.004762 test gap. Its paired
window interval is [-0.007008, -0.002514], with 563 long-context wins. The
validation interval is [-0.004280, 0.000354]. These remain conditional
window intervals, separate from the training-seed intervals below.

Five paired training seeds are registered: 7, 11, 19, 31 and 43. Primary analysis
uses paired seed differences with a 95% t interval. Shelley and Stoker supply
additional test authors; selection remains on the original validation split.
The registered practical-equivalence band is ±0.01 bits/byte. Full protocol,
raw window records, and operational bounds are in the repository directory
`experiments/subword-replication/`. The completed study also includes a matched
width/depth/context trade and one exploratory prefix-space variant.

### Recorded GPU artifact: detector positive control

The detector flags the recorded GPU `g irl` sample once. Retokenization yields
` g` (token 300), a standalone space (220), then `irl` (1009). The training
index records zero word-initial uses of `irl`. Replacing `g irl` with `girl`
in a labelled negative-control copy removes the flag. The original recorded
sample remains unchanged.

Across all 12 recorded GPU samples, the detector reports one flag. Across the
separate 96 CPU generation-grid samples, it reports zero. Those are distinct
sample sets. This establishes sensitivity on the known positive case, while
broader detector sensitivity and precision remain unmeasured.

The historical GPU samples lack original generation token IDs. The trace above
is a retokenization of the preserved text. The new training runs save original
IDs; their outputs cannot establish the token history of an older sample.
The tested seed-7 rerun sample differs from the historical sample, so no exact
replay claim is made. The tokenizer explanation remains a hypothesis.

### Per-author validation and book-clustered intervals

The saved CUDA BF16 window losses give the following breakdown, retaining all
1,024 validation windows and assigning them by target-start book:

| Author | Short bits/byte | Long bits/byte | Long minus short | Windows |
| --- | ---: | ---: | ---: | ---: |
| Hawthorne | 1.564724 | 1.566040 | +0.001316 | 249 |
| Melville | 1.717790 | 1.719611 | +0.001822 | 303 |
| Trollope | 1.449088 | 1.443248 | -0.005840 | 472 |

The observed direction differs by author. This is descriptive evidence for
these weights and passages; replicated author-specific effects remain pending.

A whole-book bootstrap resamples eight test books with replacement and retains
all windows within each drawn book. Over 20,000 draws, the test interval is
**[-0.008693, +0.000404] bits/byte**, which crosses zero. This uses all 1,024
test windows and the original -0.004762 point estimate.

As a boundary sensitivity check, exclude windows whose input context or target
crosses a book boundary: six test windows and one validation window. The test
interval becomes [-0.008785, +0.000369] over 1,018 windows; the validation
interval is [-0.005514, +0.002462] over 1,023 windows and 24 books. Both cross
zero. Raw rows preserve source book/author, losses, bytes, and exclusion flags.
Only eight test-book clusters are available. This analysis still conditions on
one test author and the trained weights, while allowing within-book dependence.

### Detectability and the registered equivalence band

The registered ±0.01 bits/byte band is a declared practical threshold. If the
five-seed interval is tightly centred near the existing -0.0048 difference,
excludes zero, and lies entirely inside that band, the reported outcome will be
**statistically detectable and practically equivalent at the registered threshold**.
Both labels can hold at once. The rule is fixed before results; the observed
seed results will determine whether its conditions are met. The threshold is
a research decision, rather than a measured threshold for writing quality.

The width/depth arm is five training seeds of one configuration: width 384,
three layers, context 256, 5,046,912 parameters. Seeds are 7, 11, 19, 31 and 43,
paired against the same five long-context training seeds. It is a width/depth/
context trade, with the token budget fixed.

### Evaluation-path precision check

Across both models and both splits, individual CPU FP32 and CUDA BF16 aggregate
scores differ by at most **0.0000525 bits/byte**. The test gap changes by
**0.0000975 bits/byte**. This supports numerical agreement of the two evaluation
paths on these checkpoints and target windows. It gives an execution check
separate from statistical uncertainty and from generation-output agreement.

The [saved review evidence](../../experiments/subword-replication/README.md#review-controls-and-book-analysis)
contains the detector controls, precision scores, book assignments, and bootstrap results.

The conservative source-name index found mixed source names in one CPU-grid
passage. Coverage and false-positive limits accompany that narrow proxy.

## Data and architecture

- Corpus: ZERO Gutenberg release `zero-gutenberg-v1-0d9254dd6a65`.
- Source: 224 Gutenberg books by 29 authors; Braid performs duplicate filtering.
- Training: 192 books, 25 authors, 85,944,815 ASCII bytes, 27,157,194 BPE tokens.
- Validation: 15,227,112 bytes, 4,807,813 BPE tokens.
- Test: 4,236,866 bytes, 1,341,450 BPE tokens.
- Tokenization: byte-level BPE, 2,048 entries, no added prefix space, lines
  retain their original newlines. Full decode round trips and exact byte counts
  were verified for all three splits.
- Decoder: 6 layers, width 256, 8 attention heads, feed-forward width 960,
  tied embeddings, rotary positions, RMS normalization, GELU, residual dropout.
- Initialization: normal standard deviation 0.02; residual output projections
  scaled by sqrt(2 * layers).
- Training: 12,208 updates, 100,007,936 token presentations per model, dropout
  0.1, AdamW, peak learning rate 0.0003, 244 warmup updates, cosine decay,
  matrix weight decay 0.01, gradient clipping 1.0.
- Compute: NVIDIA A10G, PyTorch 2.8.0, BF16 autocast with FP32 weights and AdamW
  state. The runs took about 7.0 and 7.3 minutes before final evaluation/samples.

## Samples and interpretation

The six recorded prompts include literary openings and prompts outside the
training task, such as “Fund uncertainty.” and “Who is Mara?”. Read those groups
separately: a generic prose continuation is a different outcome from answering
a question or following an instruction.

Recorded GPU samples use seed 7, temperature 0.7, top-k 40, and 128 generated
BPE tokens. Live demo generation uses CPU FP32 at the same settings, so the
precise output may differ. The [seed grid](seed-grid.json) supplies six prompts
at seeds 1–8 for both models under CPU FP32: 96 preserved outputs. This grid
supports comparison across sampling seeds; it does not replicate training.

The demo preserves raw output exactly, including `g irl` in the recorded
long-context girl-and-door sample. An earlier conversational quotation omitted
that space. The samples in this release retain it.

Both recorded “The warren is ” samples begin their continuation with
“uttered by the”. [overlap.json](overlap.json) measures case-sensitive 3- and
4-gram overlap after excluding the supplied prompt; punctuation is counted as
separate units. Shared wording is an overlap observation. A memorization claim
would require comparison to source passages and an appropriate baseline.

These models often produce plausible short phrases followed by repetitions,
contradictions, malformed words, shifts of speaker, and loss of scene. The
completed five-seed study supports practical equivalence of the context arms
at the registered threshold. Coherence remains an open evaluation question.

## Intended use and data limits

Use these checkpoints to study small language models, context length,
tokenization, and literary text continuation. Outputs can be inaccurate,
offensive, or reflect stereotypes in the source books. English historical
prose dominates the data; performance on contemporary text, other languages,
instruction following, factual questions, and long narrative consistency has
limited evidence. Inspect generated passages before reuse.

The model weights in this branch are released under the [MIT License](WEIGHTS_LICENSE.txt). Corpus
provenance and source records are in `corpus/gutenberg/` in the repository.

## Files and local demo

Weights are in `models/subword/5m-256.pt` and `models/subword/5m-1024.pt`.
`models/subword/manifest.json` records SHA-256 values for the inference exports
and their source training checkpoints. The tokenizer is frozen at
`experiments/subword-scaling/tokenizer.json`.

From the repository root, using a Python environment with `torch==2.8.0`,
`numpy==2.2.6`, and `tokenizers==0.22.0`:

```sh
python scripts/serve_subword_demo.py
```

Open `http://127.0.0.1:8765`. Prompts are processed on the local computer. The
server binds to loopback and keeps prompts out of its request logs. The static
page also supports saved-example comparison when served without Python.

```sh
python scripts/sample_subword.py models/subword/5m-1024.pt experiments/subword-scaling/tokenizer.json --prompt 'The captain looked across the sea.'
python scripts/subword_seed_grid.py
python scripts/analyze_subword_samples.py
```

## Exploratory 50M checkpoint review

The [measured 50M review](../../experiments/50m-review/README.md) reports matched-window
losses, frozen source-name counts, exact repetition, and a repetition-1.1
decoding control. The original scaling run has a separate budget from the
registered 5M study. The comparison combines parameter count, architecture,
and token budget. All sample excerpts preserve the generated text.

## Completed study and inference improvements

All 16 registered runs completed successfully in about 2 hours 12 minutes
from instance launch. The instance terminated. Estimated EC2 compute through
the finish marker is $2.21; final total billing remains pending.

The primary five-seed long-minus-short difference is +0.000060 bits/byte,
95% t interval [-0.005193, +0.005313]. The three-author mean difference is
+0.001497, interval [-0.002629, +0.005622]. Both satisfy the registered ±0.01
practical-equivalence rule, with unresolved direction.

The wider three-layer, context-256 model scores lower than the six-layer,
context-1024 model by 0.006378 bits/byte on Alcott. The long-minus-wide interval
is [0.000899, 0.011856]. Across the three selected authors, the mean advantage
is 0.008752, interval [0.005586, 0.011919]. This secondary comparison changes
width, depth, and context together. Its interval overlaps the practical band
boundary, so practical equivalence and a beyond-band gain remain unresolved.

[Saved results and inference report](../../experiments/inference-decoding/README.md)
include the complete study summary, final-checkpoint comparison, decoding
sweep, cache validation, and speed measurements.

Live generation uses a request-local KV cache. When the context window shifts,
it rebuilds the prefix so the existing reset-position behavior is preserved.
The default decoding settings remain temperature 0.7, top-k 40, penalty 1.0.
The demo offers 64, 128, or 256 new tokens. Its optional sentence-end display
uses terminal punctuation as a heuristic and retains the full raw output.
Abbreviations can affect this heuristic; an output with no detected ending
is shown in full. Saved research samples remain unchanged.
