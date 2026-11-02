# Self-editing optimizer

`self_edit.c` asks one question: what happens when a learned optimizer is
pointed at its own weights?

The target is a 1-8-1 tanh network fit to a random sine wave in 16 steps.
The editor is a 25-parameter network that works on one weight at a time.
It reads the weight and its normalized gradient and writes the change. Since
it is per-parameter, it can edit any weight vector, including its own.

1. **Bootstrap.** Normalized gradient descent trains the editor for 300 steps.
2. **Self-edit.** For 300 generations, the editor gets the meta-gradient of
   its own loss and applies it to itself. Two arms start from the same editor
   and see the same tasks:
   - `unchecked` keeps every self-edit.
   - `checked` keeps a self-edit only if loss on 32 fixed validation tasks
     does not rise.

Each row reports loss on 128 audit tasks that no decision ever sees.
Meta-gradients come from antithetic random-sign weight perturbations, so
nothing backpropagates through the inner run. A task's loss is capped at 10.

## Run

```sh
cc -O2 -std=c11 -Wall -Wextra -Wpedantic self_edit.c -o self_edit -lm
./self_edit --seed 1 > results/seed1.csv
```

One seed takes about 6 seconds on an M-series Mac.

## First results (seeds 1-5, 2026-11-02)

Audit test MSE after 16 inner steps. Lower is better.

| seed | no training | tuned SGD | editor at gen 0 | unchecked at 300 | unchecked collapse gen | checked at 300 | edits kept |
|---|---|---|---|---|---|---|---|
| 1 | 1.109 | 0.440 | 0.212 | 10 (cap) | 43 | 0.205 | 6 |
| 2 | 1.084 | 0.440 | 0.162 | 10 (cap) | 64 | 0.168 | 8 |
| 3 | 1.109 | 0.464 | 0.208 | 0.334 | never | 0.229 | 6 |
| 4 | 1.103 | 0.443 | 0.175 | 10 (cap) | 138 | 0.179 | 5 |
| 5 | 1.251 | 0.507 | 0.230 | 10 (cap) | 39 | 0.215 | 10 |

"Collapse gen" is the first generation where the unchecked editor does worse
than not training at all.

What this shows:

- The bootstrapped editor beats tuned SGD by about 2.5x on this task.
- Unchecked self-editing collapses in 4 of 5 seeds. Its weight norm grows by
  about 5 per generation, which is every one of its 25 weights moving by the
  full clip each step. The editor learned "move hard against the gradient"
  for sine-fitting weights, and that rule runs away on its own weights.
- The validation check prevents collapse in all 5 seeds, but it does not buy
  improvement. Only 5-10 of 300 edits pass. Audit loss ends within about 10%
  of the start in every seed: better in 2, worse in 3. The check works as a
  brake, not an engine, and the edits it accepts partly fit the 32 validation
  tasks rather than the task family.

## Next

- Harder, deterministic tasks in place of sine waves.
- A larger or refreshed validation set, to separate overfitting to it from
  real progress.
- Letting the editor see a feature that says whether it is editing a target
  or itself.
