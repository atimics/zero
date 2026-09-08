# Gutenberg 5M v1

This run trains ZERO's 4,852,992-parameter literary preset from scratch with seed
7, 512-character context, and the verified Gutenberg corpus from PR 23.

The plan is 100,000 updates at batch 2: 102,400,000 character presentations.
AdamW uses peak learning rate 0.0003, 1,000 warmup updates, cosine decay,
weight decay 0.01, gradient clipping at 1, and dropout 0.1. Every 500 updates,
the trainer checks 64 fixed windows from the separate validation corpus and
saves the latest checkpoint. It also saves each best validation checkpoint.

The new `--validation-text` option lets the run use all 85,944,815 training
characters. Validation comes from the separate 15,227,112-character file.
The selected model is scored once on 1,024 fixed windows from validation and
1,024 from test after training. Each final score covers 524,288 character
predictions, sampled across the named split.

`plan.json` records the exact local paths, source and binary hashes, arguments,
and sample prompts for this run. The runner checks those hashes before launch.
It writes status, checkpoints, logs, evaluation scores, samples, and the int8
export to the plan's output directory. A fresh output directory is required for
a fresh run. A stopped run needs a reviewed continuation plan because the
current cosine schedule is defined over each invocation's step count.

```sh
python3 scripts/train_gutenberg_5m.py experiments/gutenberg-5m-v1/plan.json
```

The initial local calibration measured about 5,000 characters per second with
Apple Accelerate. The planned run should take roughly five to six hours, with
variation from other work on the host. `VECLIB_MAXIMUM_THREADS=1` is set by the
runner. The launch uses `caffeinate -i` to allow training while the Mac is idle.

The export and sample comparison happen after training. The comparison uses
six fixed completion prompts for both the new model and the Warrenmind export.
The current browser showcase remains a separate deployment decision.

Validation of the trainer change:

- Sanitizer tests show that changing validation text changes its score while
  leaving the trained checkpoint bytes identical.
- Evaluation with zero training steps matches the training validation score.
- Short evaluation files produce a clear error.
- The existing native checks pass.
