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

## Run 2 (2026-11-02): square-root mix

Streams sampled by size ** 0.5, one best checkpoint per stream. Same
machine, 28.7 minutes. Files: `results/run2/`.

| | Run 1 | Run 2 |
|---|---|---|
| Blake held-out loss (best) | 1.48 | 1.34 (update 7,000) |
| Shakespeare | 1.28 | 1.16 (update 30,000) |
| Crowley | 1.38 | 1.23 (update 9,000) |
| Crownless | 0.61 | 0.56 (update 9,000) |
| Mining: picks Signal's best control | 71.4% | 86.2% |
| Dragon: matches the rule | 92.3% | 100.0% |

Dragon is solved as imitation. The open test is play: the model as the
rabbit chooser in crownless2's lineage test.

## The Shannon loop (`shannon_loop.py`)

The model proposes, the world verifies, and only verified gains train in.
Each round: 1,500 fresh flight states; the model's greedy control plus two
more sampled from its own distribution; `signal_replay` flies only those;
each state's best verified proposal becomes a training row; 150 fine-tune
updates (half of every batch replayed from the six streams). The checked arm
keeps a new model only if mean utility on 250 gate states does not fall; the
unchecked arm keeps all. 250 audit states never decide anything. Start: run
2's last checkpoint. 8 rounds, about 13 minutes per arm on an M-series GPU.
Files: `results/loop/`.

| Round | Unchecked: audit best pick | crashes | Checked: audit best pick | crashes | kept |
|---|---|---|---|---|---|
| 0 | 87.2% | 8 | 87.2% | 8 | |
| 1 | 87.6% | 9 | 87.6% | 9 | yes |
| 2 | 89.2% | 8 | 87.6% | 9 | no |
| 3 | 90.8% | 8 | 87.6% | 9 | no |
| 4 | 90.0% | 8 | 87.6% | 9 | no |
| 5 | 90.0% | 8 | 87.6% | 9 | no |
| 6 | 90.0% | 8 | 89.2% | 8 | yes |
| 7 | 88.4% | 7 | 89.2% | 8 | no |
| 8 | 88.8% | 8 | 89.2% | 8 | no |

Voice losses and dragon held steady in both arms (Shakespeare 1.157 at the
start, 1.20 after round 1, flat after; dragon 100% throughout).

What it shows:

- Both arms end a little better than they started (+1.6 and +2.0 points),
  and the unchecked arm peaked at +3.6. With 250 audit states one standard
  error is about 2 points, so none of this is yet a proven gain.
- Unchecked did not collapse, unlike the self-editing optimizer
  (`../self-edit`). Here the verifier is inside every round: Signal, not the
  model, picks each training label. The gate on top only adds a second check.
- The gate barely carries signal. Gate utility is dominated by rock crashes
  the prompt cannot describe, so it sat at -0.188 for most rounds and
  decisions turned on tiny numerical differences.
- Proposals beat the model's own greedy choice on 10-13% of fresh states each
  round, and the margin did not shrink, so the loop has not exhausted what
  the model can find with three tries.

## Next

- Weight streams by size to the 0.5 power (or cap repeats) and save the best
  checkpoint per stream instead of one by the mean.
- Put the nearest rock's bearing and distance in the mining prompt.
- Let the model play: a chooser that asks it for rabbit goals inside
  crownless2's lineage test, judged by the world, and a mining pilot judged
  by `signal_replay`.
