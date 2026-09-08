# Replication and diagnostic study

[Registered plan](PREREGISTRATION.md) defines five training seeds per context
arm, the paired window bootstrap, three test authors, a parameter-matched
width/depth trade, one prefix-space variant, and generation diagnostics.

## Existing-checkpoint window rescore

CPU FP32, 1,024 paired windows, 20,000 bootstrap draws:

| Split | Long minus short bits/byte | 95% window interval | Long wins |
| --- | ---: | --- | ---: |
| Validation | -0.001960 | [-0.004273, 0.000355] | 539 / 1,024 |
| Test | -0.004859 | [-0.007108, -0.002611] | 560 / 1,024 |

This rescore uses CPU FP32; the original CUDA BF16 test gap was -0.004762.
The window interval is conditional on the two trained checkpoints. Correlated
windows within books, training-seed variation, and new-author variation remain
outside this interval. The GPU replication saves per-window losses too.

## Diagnostic baseline

The 96 CPU FP32 samples produced zero word-initial tokens that were never
word-initial in training. This criterion has zero observed flags on
that grid. Retokenizing decoded
samples can differ from their original generation IDs; the new runs save IDs.

A conservative source-name index found multiple source books in one of the 48
short-context passages and zero of the 48 long-context passages. This is a
sparse heuristic. Names can be ambiguous and most words have no unique book
association. The raw records include matches and coverage. This is separate
from a validated coherence measure.

## Data and engineering

Shelley and Stoker are new test authors, absent from the original author list.
Their 1,024 windows per book passed the registered exact 64-word overlap check.
Source and clean hashes are frozen in the author manifest. The prefix-space
variant is fitted only on training text; the manifest records the added spaces.

Model weights are MIT-licensed in `models/LICENSE`. Engineering checks cover
pairing, interval calculations, parameter matching, and seed controls. These
checks verify calculations and execution; the registered statistical analysis
will determine the experimental conclusion.

## CUDA BF16 confirmation

The GPU rescore reproduces the original test gap: -0.004761776 bits/byte.
Its paired-window 95% interval is [-0.007007829, -0.002514086], with 563
long-context wins out of 1,024 windows. Validation has an interval spanning
zero: [-0.004280394, 0.000354189]. These are conditional window intervals;
the five training-seed pairs are running.


## Review controls and book analysis

The detector catches the historical GPU `g irl` positive control. Its repaired
`girl` control has zero flags. The 12 recorded GPU samples contain one flag;
the separate 96 CPU samples contain zero. Historical token traces use
retokenization. New runs save generation IDs. See
[gpu-sample-detector-control.json](gpu-sample-detector-control.json).

The whole-book bootstrap of all 1,024 test windows gives a 95% interval of
[-0.008693, +0.000404] bits/byte. This crosses zero. Per-author validation gaps
are +0.001316 for Hawthorne, +0.001822 for Melville, and -0.005840 for Trollope.
[Book analysis](book-cluster-analysis.json) includes source hashes, raw rows,
per-author scores, and a sensitivity check excluding book-boundary windows.
Reproduce it with the frozen corpus, token data, and saved CUDA window losses:

```sh
python scripts/analyze_book_clusters.py --ready CORPUS_READY --tokens TOKEN_DATA --losses SAVED_RESCORES --output book-cluster-analysis.json
```

The loss directory contains `existing-window-rescore-5m-256-windows.json` and
`existing-window-rescore-5m-1024-windows.json`. The analysis verifies corpus
ordering and byte denominators before resampling whole books.

The [evaluation-path precision check](eval-precision-check.json) finds at most
0.0000525 bits/byte difference between individual CPU FP32 and CUDA BF16 scores.
The test gap changes by 0.0000975 bits/byte.

The width/depth arm runs five seeds of one configuration: width 384, three
layers, context 256. The same five seeds pair with the long-context arm.
Under the registered ±0.01 band, a tight interval near -0.0048 that excludes
zero will be labelled statistically detectable and practically equivalent.
