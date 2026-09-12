# Inference and decoding study

The cache preserves the existing sliding-context semantics by rebuilding
when the prefix shifts. Original uncached sampling remains the default for
research callers; the live demo explicitly enables the cache.

A fixed decoding grid scores 12 settings: temperatures 0.6/0.7/0.8, top-k
20/40, and repetition penalties 1.0/1.1. Each setting uses the six fixed
prompts and generation seeds 1–8, for 576 outputs from the selected 50M
checkpoint. Every continuation has 128 new tokens. Repetition and diversity
are descriptive measures; blind coherence ratings require separate review.

The completed 16-run replication summary and successful finish record are
included here. The model card reports the paired training-seed intervals.

The demo offers 64/128/256 new tokens and optional display trimming at the
last detected sentence ending. This is a punctuation heuristic. It retains
full raw output, which the user can reveal with a checkbox. Samples with no
detected ending are displayed in full.

## Best versus final 50M checkpoint

Both checkpoints were rescored in CPU FP32 on the same 1,024 windows per
split, with identical starts and byte counts checked explicitly.

| Checkpoint | Validation bits/byte | Test bits/byte |
| --- | ---: | ---: |
| Best, step 35,500 | 1.340900 | 1.391370 |
| Final, step 97,657 | 1.403647 | 1.453993 |

Final test loss is 0.062622 bits/byte higher. Keep the selected best checkpoint
for generation. This comparison conditions on these weights and held-out
passages. `checkpoint-comparison.json` and `final-50m-windows.json` preserve
the aggregate scores, raw losses, and checkpoint hash. The training job's
CUDA BF16 rescore of the best checkpoint also agrees closely with CPU FP32.

## Decoding sweep

All 576 outputs completed. Each setting uses six prompts and eight seeds.
The `sweep` directory contains the manifest, raw output and token IDs, full
metrics by prompt, and a review packet. These are generation seeds at fixed
weights. The sweep uses cached CPU inference, whose separate agreement check
is reported below.

| Temperature | Top-k | Repetition | Bigram repeat excess | Four-gram repeat excess |
| ---: | ---: | ---: | ---: | ---: |
| 0.6 | 20 | 1.0 | 22.03% | 6.25% |
| 0.6 | 20 | 1.1 | 19.83% | 5.37% |
| 0.6 | 40 | 1.0 | 18.40% | 5.07% |
| 0.6 | 40 | 1.1 | 17.80% | 4.55% |
| 0.7 | 20 | 1.0 | 16.00% | 3.03% |
| 0.7 | 20 | 1.1 | 16.49% | 3.72% |
| 0.7 | 40 | 1.0 | 12.86% | 2.41% |
| 0.7 | 40 | 1.1 | 11.56% | 1.68% |
| 0.8 | 20 | 1.0 | 10.24% | 0.89% |
| 0.8 | 20 | 1.1 | 8.62% | 0.76% |
| 0.8 | 40 | 1.0 | 8.45% | 0.55% |
| 0.8 | 40 | 1.1 | 8.23% | 0.74% |

Metrics count repeated occurrences beyond an n-gram's first appearance,
divided by all n-gram positions in the generated continuation. They ignore
case and punctuation. Means weight each prompt/seed equally.

At temperature 0.7 and top-k 40, penalty 1.1 lowers bigram repeat excess from
12.86% to 11.56%, and four-gram excess from 2.41% to 1.68%. Bigram repetition
falls on four of six prompts; it rises on the fairy-tale and Mara prompts.
The average across-seed distinct-bigram ratio rises from 0.7971 to 0.8052.
Average generated word count rises from 77.69 to 78.73 under the fixed
128-token budget. Per-prompt values and lengths are available in the summary.

Higher-temperature settings often repeat less. A quality decision also needs
consistency ratings. The default remains temperature 0.7, top-k 40, penalty
1.0. The demo exposes penalty 1.1 as an optional control.

## Blind consistency review

[Review packet](sweep/blind-review.md) contains 48 paired cases (96
continuations) comparing penalty 1.0 and 1.1 at temperature 0.7/top-k 40.
The pair was chosen from the earlier control, independently of this grid's
scores. Prompt order and A/B assignment use a fixed random seed. The settings
key is stored separately from the review page.

The standalone `sweep/blind-review.html` offers ratings from 1 to 5 and a JSON
export. The rubric asks about consistent participants, setting, and actions.
Human ratings are pending. These files provide the review materials; the
repetition and diversity scores above are separate measurements.

## Reproduce

```sh
python scripts/run_decoding_sweep.py --checkpoint BEST_50M --tokenizer experiments/subword-scaling/tokenizer.json --output experiments/inference-decoding/sweep
python scripts/summarize_decoding_sweep.py --directory experiments/inference-decoding/sweep
python scripts/benchmark_cached_inference.py --large-checkpoint BEST_50M --models models/subword --tokenizer experiments/subword-scaling/tokenizer.json --output experiments/inference-decoding/cache-benchmark.json
python scripts/score_single_subword.py --checkpoint FINAL_50M --data TOKEN_DATA --output experiments/inference-decoding/final-50m-windows.json
```

## Cache agreement and local speed

Trained checkpoints were checked on prefill and 16 incremental steps.
All 18 prompt/model samples have identical generated token IDs between the
cached and uncached paths. Both timing cases also preserve IDs for all models.
Maximum observed absolute logit differences are 0.00000716 (5M/256),
0.00000668 (5M/1024), and 0.00002480 (50M). These are measured FP32 agreement
checks, with tolerance 0.0001; other prompts can encounter numerical ties.

Timings use CPU FP32, two threads, three alternating-order repetitions, and
median end-to-end generation time including prefill. Hardware/software details
and every timing are saved in `cache-benchmark.json`. These are local timings.

| Model | Short prompt, 128 new tokens | Near context limit, 8 new tokens |
| --- | ---: | ---: |
| 5m-256 | 3.43× | 1.83× |
| 5m-1024 | 3.26× | 1.88× |
| 50m | 4.30× | 1.99× |

The near-limit case starts four tokens below the model's context limit. Cache
rebuilds preserve the original context-window behavior and reduce the speed
gain there. The short prompt is the fixed girl-door opening. These checks
measure execution and speed; model quality remains a separate result.

Validation also includes cached chunk masks, context overflow, independent
request caches, C-style repetition math, bounded generation options, raw-text
recovery, and the existing C/PyTorch parity and subword tests. Browser checks
covered live generation, 64-token output, repetition 1.1, sentence trimming,
and recovery of the full generated text.
