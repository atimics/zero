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
word-initial in training. This criterion has zero observed baseline errors on
that grid. It cannot support an improvement claim there. Retokenizing decoded
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
