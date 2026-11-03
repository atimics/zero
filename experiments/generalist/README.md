# Generalist: four voices, two worlds

One ZERO literary model (4,852,992 parameters, 512-byte context) trained on
six tagged streams at equal weight:

| Stream | Source | Train size |
|---|---|---|
| `<blake>` | `corpus/blake.txt` | 59 KiB |
| `<shakespeare>` | `corpus/shakespeare.txt` | 5.0 MiB |
| `<crowley>` | `corpus/crowley.txt` | 264 KiB |
| `<crownless>` | Crownless held-account corpus, spoken lines from `src/story`, Hra'khor corpus | 104 KiB |
| `<mining>` | Signal flight states; label = the best of 9 controls over 1 s by `signal_replay`'s own `utility` | 6,000 rows |
| `<dragon>` | crownless2 rabbit goal choices from `cc2trace` (1 in 100), 16 worlds x 1 year | 45,129 rows |

World validation rows use seeds the training rows never touch. A third
world (gacha) is not on disk yet.

## Run

```sh
python3 build_data.py --signal-replay ../../../signal/build/signal_replay \
  --cc2trace ../../../crownless2/build/cc2trace --crownless ../../../crownless
cc -O2 -std=c11 ../../tests/torch_reference.c -o /tmp/zero-init -lm && /tmp/zero-init initialize initial.ckpt
MARKET=on-demand INSTANCE_TYPES=g5.xlarge ./train_aws.sh
```

`train_aws.sh` starts one GPU instance with a throwaway key and security
group, powers off after `MAX_MINUTES`, copies `output/` to `runs/`, and
deletes everything on exit. The account's spot GPU quota is 0, so use
on-demand.

## First run (2026-11-02)

g5.xlarge (A10G), 30,000 updates x 16 windows, 26.5 minutes of training,
about $0.60 of EC2. Files: `results/`.

**The worlds.** Scored on the step-2,000 checkpoint (see below).

- Dragon: the model's goal matches the rule's in **92.3%** of 3,000 held-out
  choices. Always choosing the most common goal (graze) scores 46.1%.
- Mining: the model picks Signal's best control in **71.4%** of 500 held-out
  states. Median regret is 0. Mean utility is still worse than a random
  control (-0.238 vs -0.155), because in 5 of 500 states it flies into rock
  (utility about -11 to -15, against a usual range of about ±0.1). The prompt
  has no hazard information, so the model cannot see the rock. That is a
  data gap, not a learning failure.

**The voices.** Held-out loss (nats per byte) by update:

| Update | Blake | Shakespeare | Crowley | Crownless |
|---|---|---|---|---|
| 1,000 | 1.66 | 1.79 | 1.55 | 1.07 |
| 4,000 | 1.92 | 1.46 | **1.36** | **0.60** |
| 30,000 | 3.15 | **1.28** | 2.28 | 0.90 |

Equal stream weight shows Blake's 59 KiB about 700 times, and it is
memorised by update 2,000. Crowley follows by update 5,000 and Crownless by
5,000. Shakespeare keeps improving to the end. Selecting the checkpoint by
mean held-out loss therefore picked update 2,000, which is undertrained for
Shakespeare and both worlds. Samples (`results/samples.json`) come from that
checkpoint; each tag produces its own register, but the words are not yet
real English.

## Next

- Weight streams by size to the 0.5 power (or cap repeats) and save the best
  checkpoint per stream instead of one by the mean.
- Put the nearest rock's bearing and distance in the mining prompt.
- Let the model play: a chooser that asks it for rabbit goals inside
  crownless2's lineage test, judged by the world, and a mining pilot judged
  by `signal_replay`.
