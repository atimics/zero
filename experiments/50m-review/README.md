# Measured review of the 50M checkpoint

This is an exploratory review of the saved step-35,500 checkpoint from the
original scaling run. The registered study covers 5M context pairs, a 5M
width/depth/context trade, and a prefix-space variant. Its operational section
explicitly places the 50M run under its separate original limit.

## Matched scoring

The scorer uses the same 1,024 starts, 64 target tokens per window, tokenizer,
and byte denominators as the published 5M evaluation. All scores in the
comparison use CPU FP32. Raw losses are preserved in `windows-50m.json`.
All start-token and byte-count pairs were checked against both 5M files.

| Model | Validation bits/byte | Test bits/byte |
| --- | ---: | ---: |
| 5M, context 256 | 1.552207 | 1.568528 |
| 5M, context 1,024 | 1.550247 | 1.563669 |
| 50M, context 1,024, step 35,500 | 1.340900 | 1.391370 |

The test reduction versus 5M/1,024 is 0.172299 bits/byte (11.02%). Validation
uses 206,379 target bytes; test uses 207,306. This supports lower prediction
loss for this checkpoint under the combined changes below. These point
estimates condition on the selected weights and passages.

The comparison measures the combined change: 5M to 50M parameters, width 256
to 640, six to ten layers, feed-forward width 960 to 2560, and a larger training
budget. Both long-context models use context 1,024. The 50M checkpoint has seen
290,816,000 token presentations (35,500 updates). The 5M runs have a
100,007,936-token budget (12,208 updates), with earlier best-checkpoint selection.
Checkpoint selection uses the same 64 validation windows. Test windows are
scored after selection. This remains one training seed per original model.

## Names: coverage limits matter

With the frozen unique-book index, the mean distinct source books per passage
across six CPU samples is 0.000 for 5M/256, 0.167 for 5M/1024, and 0.000 for 50M.
All three sets have zero passages with multiple matched source books. The
recorded historical GPU sets give 0.000 and 0.333 for the two 5M models; the
latter includes one passage with two matched source books. Those historical
outputs are labelled separately from the CPU comparison.

Sara is absent from the frozen index. The training source contains internal
capitalized occurrences in six books: Gutenberg 137 (165), 146 (630), and
86, 145, 6308, 7469 (one each). The first two are *Sara Crewe* and *A Little
Princess*. A strict unique-book rule excludes a character shared by versions
of the same story. The zero therefore includes a coverage failure.
These counts provide no established three-to-one improvement or validated
coherence gain. The index stays frozen for this analysis.

## Exact repetition

Metrics use the full generated continuation, excluding the prompt, lowercase
words, and n=1,2,3,4. Punctuation is ignored. For each n, max frequency is the
largest n-gram count divided by the number of positions. Repeat excess is the
sum of occurrences beyond each n-gram's first occurrence, divided by positions.
Lengths are recorded. Exact repetition captures lexical reuse; paraphrases
and the usefulness of a repeated phrase require other measures.

| 50M prompt | Most frequent bigram | Count / positions | Bigram repeat excess |
| --- | --- | ---: | ---: |
| Girl | to see | 4 / 78 | 12.82% |
| Captain | to say | 2 / 86 | 3.49% |
| Fund uncertainty | french french | 9 / 71 | 33.80% |

All six prompts at each scale appear in `sample-diagnostics.json`. These are
one generation seed per prompt. The 50M sample set includes more exact
repetition on several prompts; aggregate writing-quality conclusions need
broader evidence.

## Repetition 1.1 control

The same checkpoint, CPU FP32, seed 7, temperature 0.7, top-k 40, and 128 new
tokens were used for both outputs. The control applies the C sampler's rule:
divide each distinct token's probability by 1.1 if it occurs in the last 64
tokens, before temperature and top-k. In logits this is subtracting log(1.1).
The penalty-1.0 output exactly reproduces the previously shown French sample.

Bigram repeat excess falls from 33.80% to 2.50%; four-gram repeat excess falls
from 7.25% to zero. The French loop disappears in this draw. The replacement
still contains a telegram, a lantern, a Spaniard playing ball with his horse,
and repeated trees. This shows sensitivity to decoding settings in one seed.
It leaves semantic coherence as a separate question. Both raw outputs and
original token IDs appear in `repetition-control.json`.

## Corrected caption

The shown girl excerpt keeps a room and two participants for four lines, then
ends with a repeated question awaiting a reply. The full saved continuation
includes a reply and further dialogue. The excerpt supports that specific
observation.

## Reproduce

```sh
python scripts/sample_scaling_review.py --large-checkpoint BEST_50M --models models/subword --tokenizer experiments/subword-scaling/tokenizer.json --output experiments/50m-review
python scripts/score_single_subword.py --checkpoint BEST_50M --data TOKEN_DATA --output experiments/50m-review/windows-50m.json
python scripts/measure_sample_failures.py --directory experiments/50m-review --names experiments/50m-review/source-names.json --historical docs/subword/results.json
```

The checkpoint hash is recorded in both the score file and the saved 50M
sample file. `sample()` accepts `repetition_penalty=1.1`; its default is 1.0.
The probability-division equivalence test covers the last-64-token rule.

## Queue status at 2026-09-08 21:05 UTC

The replication instance remains running. Saved training results: 15/16.
Elapsed instance time is 2.10 hours, with estimated EC2 compute
of $2.11 at the package rate. The four-hour maximum and $5 target
apply to this instance. Completion markers and final billed total remain
pending. Storage and transfer charges are separate from this compute estimate.
The separate original 50M run has a 12-hour maximum and $14 budget target.
The source snapshot is saved in `queue-status.json`.
