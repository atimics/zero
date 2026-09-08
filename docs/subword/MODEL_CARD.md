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
short-context model, a 0.30% reduction. This is a small measured prediction
gain in one training seed. It supplies limited evidence about general effects
of context length. Sample pairs support observations about those particular
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

Validation curves record 123 checks for each model. Their raw histories are
in [results.json](results.json), and [curves.svg](curves.svg) shows the full
selection history. The test split is scored after checkpoint selection.
The experiments use one training seed and one held-out test author. The scores
have no reported confidence interval or multi-seed training replication.

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
current evidence supports a small held-out prediction improvement from longer
context, while coherence remains an open evaluation question.

## Intended use and data limits

Use these checkpoints to study small language models, context length,
tokenization, and literary text continuation. Outputs can be inaccurate,
offensive, or reflect stereotypes in the source books. English historical
prose dominates the data; performance on contemporary text, other languages,
instruction following, factual questions, and long narrative consistency has
limited evidence. Inspect generated passages before reuse.

The repository has no explicit model license at this release revision. Corpus
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
