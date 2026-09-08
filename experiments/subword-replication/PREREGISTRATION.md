# ZERO 5M replication: registered plan

Registered before the new training runs and before scoring the new test authors.
The existing one-seed aggregate results and raw samples were already known.
The existing checkpoint window rescore was started before this registration;
its results had not been inspected when this document was written.

## Claim and prediction

Current claim: We observed a 0.30% difference in one training seed. Its size
relative to run-to-run variation remains unknown. Multi-seed replication is
pending.

Prediction: At 100,007,936 token presentations, the mean long-minus-short test
bits/byte difference will be small, with practical equivalence defined in
advance as a difference within ±0.01 bits/byte. Both positive and unresolved
outcomes will be reported with individual seed results.

## Primary experiment

Train five matched pairs from scratch: seeds 7, 11, 19, 31, and 43. Each pair
uses identical initial weights and training target blocks, with the existing
5,049,600-parameter configurations at context 256 and 1,024. Sampling blocks
use seed + 64, separately from dropout. All other settings match the original
100M-token experiment. Seed 7 is rerun as a reproducibility check and counted
once in the five-pair analysis.

Select checkpoints on the same 64 validation windows, every 100 updates and at
the final update. Score the same 1,024 Alcott test windows. Primary statistic:
mean paired long-minus-short test bits/byte across five training seeds. Report
all five deltas, their sample standard deviation, and a two-sided Student-t
95% interval with 4 degrees of freedom. An interval entirely below zero is
evidence of lower prediction loss in this setup; an interval containing zero
leaves the direction unresolved. An interval entirely inside ±0.01 supports
practical equivalence at the registered threshold. Report both decisions.

A window bootstrap is secondary and conditional on fixed weights. Resample
paired windows 20,000 times with seed 1701; each draw uses summed loss divided
by summed bytes. Report a 95% percentile interval and window win counts.
Within-book dependence and training-seed uncertainty remain outside this
window-only interval. A CPU FP32 rescore is labelled separately from the
original CUDA BF16 evaluation; CUDA BF16 replication records all window losses.

## Additional test authors

Keep training and selection splits unchanged. Add Mary Shelley (Gutenberg 84,
Frankenstein) and Bram Stoker (Gutenberg 345, Dracula). Neither author appears
in the existing 29-author corpus. Freeze downloaded hashes and cleaning before
scoring. Check cleaned 64-word spans for overlap with training, report overlap,
and exclude matching evaluation windows. Score 1,024 fixed windows per author.
Report individual author scores and the unweighted mean across Alcott,
Shelley, and Stoker. These are three selected authors, with one book each for
the two additions; broad English generalization remains an open question.

## Width versus context

Train width 384, three layers, feed-forward width 1,080, 12 heads, context 256
using the same five seeds and 100M-token budget. This gives 5,046,912 parameters
versus the existing width-256, six-layer model's 5,049,600 (0.053% difference).
The comparison trades depth and context for width. It is not a pure width
intervention. Use the same paired analysis against the long-context arm and
report this as secondary, with no adjustment of the primary decision.

## Prefix-space variant

Train one exploratory seed-7 short-context model using a newly fitted BPE with
prefix spaces enabled, on training text only. Added line-prefix spaces are
recorded as preprocessing. Compare held-out raw byte targets through a common
raw-text scoring path; document any target-boundary approximation. Compare
formatting metrics on the fixed six-prompt, eight-generation-seed grid before
and after. This single exploratory variant is not a replicated causal result.
The proposed tokenizer explanation for `g irl` is a hypothesis.

## Generation diagnostics

Use six fixed prompts and generation seeds 1–8, 128 tokens, temperature 0.7,
top-k 40. Preserve token IDs and decoded text. Count generated word-initial
alphabetic tokens that were never observed word-initially in training. Report
rates, counts, and examples; retokenized historical samples are labelled.
This diagnostic measures unusual token placement, not grammatical correctness.

Build a source-name index only from training books. Use capitalized words
inside sentences, at least five occurrences, associated with exactly one book.
Exclude a frozen common-word list. For each generated passage report matched
names, source book IDs, coverage, and distinct source count. Source mixing is
a proxy with false positives and missed names, not a complete coherence metric.
No model judge or manual scoring is used for these diagnostic counts.

## Operational bounds and reporting

Use one additional g5.xlarge for a maximum of four hours, budget target $5,
with periodic uploads and termination on shutdown. The existing 50M experiment
continues under its separate original limit. A calibration failure preserves
completed evidence and is reported. Statistical results, missing runs, and
engineering/CI checks appear as separate sections. This plan will be committed
and pushed before dispatch. Subsequent changes are recorded as amendments.

### Amendment 1: exact byte boundaries for the prefix comparison

Before running the variant, specify shared-boundary trimming: encode each
canonical prefix-plus-target with both tokenizers, trim the target edges to
boundaries shared by both encodings, and score only full tokens covering those
same bytes. Report trimmed byte counts. Added line-prefix spaces remain outside
targets. This avoids fractional token-loss allocation at a byte boundary.
