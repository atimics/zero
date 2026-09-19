# Plan: put the stance on a labelled line

Status: proposed · 2026-09-16 · target `crownless.moves.typed-stance.v1`

## The problem, measured

The shipped checkpoint (`ed2e0316…`, 47/48 move accuracy) performs all twelve
conversational moves and reads almost nothing of the speaker's stance. Every number
below is from held-out rows of `crownless-moves-v2` against that checkpoint.

**Intervention — change one stance line, see whether the reply moves at all (120 rows each):**

| line | values | reply changes | corpus leverage |
|---|---:|---:|---:|
| `# voice:` | 9 | 42% | 0.447 |
| `# goal:` | 4 | 20% | 0.432 |
| `# stress:` | 3 | 8% | 0.459 |
| `# courage:` | 3 | 8% | 0.422 |

Corpus leverage is how far the target text moves when only that axis changes. It is flat
across all four axes, and `stress` has the *highest* leverage with the joint-lowest
effect. `stress` and `courage` hold cardinality fixed and vary leverage: identical effect.
`voice` against `stress` varies cardinality: five times the effect. **The effect tracks
cardinality, not how much the answer actually depends on the line.**

**Five explanations are falsified and must not be re-tested:**

1. Tokenisation — `low` / `medium` / `high` produce plainly distinct token ids.
2. Coverage — 58% of (move, voice, rule, variant) groups appear at all three stress levels.
3. Lexical separability — stress levels share 0.71 of their vocabulary, voices 0.66.
4. Position — `# voice:` sits *furthest* from the generated text and is the strongest line.
5. Gradient magnitude — targets differing only by stress are as far apart textually as
   those differing only by voice (0.463 against 0.435).

**The cost of the failure:** cell accuracy is capped at 20/48 on the guard, and `dispute`
appears to be the weakest move only because a dispute wording is a cross-product of
stress-specific opener, copied account and stress-specific challenge — 10 of its 12 misses
take the opener from the wrong stress, scoring zero twice while being a perfectly good
dispute. Credit those and move accuracy is 215/240 rather than 203.

## What this is called

Gradient starvation: cross-entropy is minimised by capturing a subset of the predictive
features, and the captured features suppress the learning dynamics of the rest
([Pezeshki et al., NeurIPS 2021](https://arxiv.org/abs/2011.09468)). The older sibling is
posterior collapse, where a decoder able to model the data without the latent ignores it
([Bowman et al., CoNLL 2016](https://aclanthology.org/K16-1002/)).

## Why not the cheaper fixes

**Inference-time guidance is already implemented and measured**
(`scripts/guided_crownless_moves.py`, following
[Ho & Salimans 2022](https://arxiv.org/abs/2207.12598) and
[Sanchez et al., ICML 2024](https://arxiv.org/abs/2306.17806) for pure language models).
Over 120 rows it lifts cell accuracy from 47 to 51 at weight 2 with no move cost, and to
54 at weight 8 where it starts costing moves. **Five points, ceiling 45%.** A feature the
model had learned and merely underweighted would recover far more; five points is what it
looks like when there is little there to amplify. Keep it as a knob, not as the fix.

**An auxiliary discriminator** ([Hu et al., ICML 2017](https://arxiv.org/abs/1703.00955))
would force stance into the hidden state, but it adds a training-time objective to defend
a signal that should not have been competing for attention in the first place, and leaves
the 39 tokens of prompt text in place.

**More steps** is refuted: the 32,000-step run plateaued exactly where the 8,000-step run did.

## The change

The model already has two conditioning pathways, and only one of them fails.

`Crownless.hidden` adds typed embeddings — role, knowledge, provenance, event, event kind —
directly into the residual stream at every position, bypassing attention entirely. Sorted
by pathway:

| carried on | signal | result |
|---|---|---|
| meta channel | field role, knowledge, provenance | copy retention **12/12** |
| meta channel | event kind | move accuracy **47/48** |
| prompt text | voice, goal, stress, courage | read 42% / 20% / 8% / 8% |

Everything on the typed channel works. The stance is the only conditioning the model must
retrieve by attending back over text, and it costs **39 of 98 prefix tokens — 44% of the
average prompt, up to 82%** — to be ignored.

This is the fly's labelled line: an olfactory receptor type converges on one glomerulus, so
identity is carried by *which wire is active* rather than by a pattern decoded from a shared
bus. The same discipline shows up in
[Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3), where
fixing connectivity by cell type and leaving only per-neuron parameters free predicted
measured activity across 26 studies — typing the units is what buys the statistical
efficiency. (Both are design analogies, not biology: this model is trained end to end by
gradient descent and has no developmental wiring.)

### Files, in dependency order

**1. `zero/scripts/crownless_v2.py` — the encoder and the model**

- `Config`: add `voices: int = 16`, `goals: int = 8`, `stresses: int = 4`, `courages: int = 4`.
- `Crownless.__init__`: four `nn.Embedding` tables, inserted **after** `kinds` so existing
  tensor order is preserved and only new entries append to the export layout.
- `Crownless.hidden`: add the four, gated on `meta[..., 5] != 0`, applied at every position
  rather than only where a field sits — that is the whole point.
- `encode_row`: stop emitting `# voice:`, `# goal:`, `# stress:`, `# courage:` as text.
  Widen `meta` from 5 fields to 9 and fill 5–8 across the whole prefix.
- Keep `# memory:` and `# thought:` as text. They are content, not categories.

Parameter cost: (16 + 8 + 4 + 4) × 192 = **6,144**, or 0.12% of the model.

**2. `zero/scripts/build_crownless_moves.py`** — emit the stance as row fields only; the
prefix no longer carries the four lines. The `accepted` pools and every other axis are
untouched, so the metric is unchanged and the comparison against `ed2e0316…` stays honest.

**3. `zero/scripts/train_crownless_moves.py`** — the bridge anchor stays exactly as it is.
This run changes one thing, as the last one did.

**4. `crownless/src/story/cc_core_model.c`** — the runtime must build the identical prompt.
`Hidden()` mirrors the Python addition: `weights[1..5]` become `weights[1..9]`, the block
base moves from 6 to 10, `TENSORS` 59 → 63, and `m->meta[i]` widens from 5 to 9.
`CcCoreModelBeginMind` stops writing the four stance lines and sets the meta fields instead.

**5. `crownless/tools/language/compile_model.py`** and the reference cases — regenerate.
`CORE_LAYOUT` is generated from the export header, so the new tensors appear automatically;
`TENSORS` and the block base are the two hand-maintained constants that must move with them.

### The parity rule

The Python encoder and the C runtime must produce byte-identical prompts. This session lost
a morning to a runtime that emitted cue strings its weights had never seen, and an afternoon
to the Python encoder defaulting a missing voice to `resident` while the C omitted the line.
Both were found by `core_model_native_parity` comparing 143 sentences.

**Before the training run:** extend `tools/core_model_probe.c --dump-meta` coverage to the
new fields and add a parity case with a non-default stance on every axis. A meta channel is
harder to eyeball than a text line — a wrong field id is invisible in the decoded prompt —
so the dump is the only way this stays honest.

## How we will know, declared before the run

Same corpus, same base checkpoint, same 8,000 steps, same bridge anchor, same 48-row guard,
same metric. One variable.

| metric | shipped | gate |
|---|---:|---|
| move accuracy | 47/48 | **≥ 40/48** — must not regress |
| copy retention | 12/12 | **12/12** — hard |
| degenerate / leaked | 0 / 0 | **0 / 0** — hard |
| variety (thinnest move) | 2 | **≥ 2** — hard |
| **stress intervention rate** | **8%** | **≥ 30%** — the point of the run |
| cell accuracy | 20/48 | recorded, not gated |

The intervention rate is the primary metric because it measures the thing that failed, and
it cannot be satisfied by a model that merely memorises more wordings. Thresholds are fixed
here, before the run, and are not moved during it.

Secondary, free: the prompt should shorten by about 39 tokens, which is worth measuring
because context is the scarcest resource in a 512-token window.

## Risks

- **Export format change.** New tensors shift the native layout. The `_Static_assert` on
  `TENSORS` catches a mismatch at compile time; the file-hash pin catches a stale asset at
  load time. Both already exist.
- **The typed channel could over-condition**, collapsing wording variety within a cell. The
  variety floor is in the gate for exactly this.
- **Memories and thoughts stay textual**, so the prefix does not become uniform; if the
  model starts ignoring *those*, the same measurement applies and the same fix is available.
- **Rollback** is the shipped checkpoint and a `git revert`; nothing here is one-way.

## Citations

- Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie. *Gradient Starvation: A Learning
  Proclivity in Neural Networks.* NeurIPS 2021. https://arxiv.org/abs/2011.09468
- Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio. *Generating Sentences from a Continuous
  Space.* CoNLL 2016. https://aclanthology.org/K16-1002/
- Hu, Yang, Liang, Salakhutdinov, Xing. *Toward Controlled Generation of Text.* ICML 2017.
  https://arxiv.org/abs/1703.00955
- Ho, Salimans. *Classifier-Free Diffusion Guidance.* 2022. https://arxiv.org/abs/2207.12598
- Sanchez, Fan, et al. *Stay on Topic with Classifier-Free Guidance.* ICML 2024.
  https://arxiv.org/abs/2306.17806
- Lappalainen et al. *Connectome-constrained networks predict neural activity across the fly
  visual system.* Nature 2024. https://www.nature.com/articles/s41586-024-07939-3
- Dasgupta, Stevens, Navlakha. *A neural algorithm for a fundamental computing problem.*
  Science 2017. https://www.science.org/doi/10.1126/science.aam9868
- Dorkenwald et al. *Neuronal wiring diagram of an adult brain.* Nature 2024.
  https://www.nature.com/articles/s41586-024-07558-y
