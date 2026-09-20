# Plan: the next generation of the crownless core

Status: proposed · 2026-09-16 · successor to `PLAN-typed-stance.md`

One sentence: **this model's speech is exactly as good as its conditioning, and its
cheapest capability is copying.** Everything here follows from those two measurements.

## What is established, with numbers

| finding | evidence |
|---|---|
| A distillation premise that names no behaviour is a trap | anchor on spoken-move rows held the student to a teacher that performs none of them, over 41,996 of 41,996 output tokens; moving it to bridge rows took move accuracy 19/48 → 47/48 at identical corpus, base, budget and metric |
| Conditioning the model cannot read is decoration | stance as prompt text: intervention 8%; as typed meta ids: **61%**. Guard cell accuracy 20/48 → 44/48, test 120/260 → 230/260, for 6,144 parameters |
| Feature learning is governed by cardinality | voice (9 values) 42%, goal (4) 20%, stress (3) 8%, courage (3) 8%, while corpus leverage is flat at 0.42–0.46. Gradient starvation, [Pezeshki et al. NeurIPS 2021](https://arxiv.org/abs/2011.09468) |
| Steps are not the lever | move ceiling reached at step 6,000 of 8,000; the 32,000-step run plateaued 15–21/48 across five epochs and validation turned at 3.7 |
| Corpus expansion is backwards for the failing moves | `settle` has 45 wordings shown 185× each and scores 2/20; `answer` has 4,186 shown twice and scores 17/20 |
| Inference-time guidance cannot substitute | classifier-free guidance over the stance buys +5 points and saturates; there was little to amplify |
| Copying is the strongest thing this model does | copy retention **60/60** on test, through every configuration tried |
| The conversation needed a policy, not a better model | cueing `answer` forever gave 5 distinct lines in 6 turns with two malformed; the corpus's own transition table gave 6/6 coherent, with no retraining |
| Characters are given almost nothing to be specific about | 50% of training rows carry neither a memory nor a thought, while the sim holds hunger, shelter, debt, trust, travel, lineage and faction |

## Phase 1 — Ship what is already measured

The typed-stance checkpoint (`c0d48351…`) is accepted and unshipped.

- **Runtime**: `TENSORS` 59 → 63, block base 6 → 10, `m->meta` widened 5 → 9,
  `CcCoreModelBeginMind` writes stance ids instead of four lines, and a voice
  name → id table that matches `crownless_v2.VOICE_IDS` exactly. The id order is a
  wire format; reordering it silently corrupts every prompt.
- **Assets**: `compile_model.py` re-pin, `CORE_LAYOUT` regenerate, reference cases
  regenerated with `typed_stance=True`.
- **Then** widen the stance to the character's situation: `hungry`, `sheltered`,
  `in_transit`, `far_from_home`, `owes_listener`, `trusts_listener`, `faction`.
  All are computed by the sim already. ~10k parameters.

*Gate: every new axis ≥30% intervention individually; move accuracy ≥ 40/48; copies 12/12.*

**Known risk, already observed:** typing stress dropped `goal` from 20% to 16%. Typed
channels compete for finite capacity. Seven more axes may starve one another, so the
gate is per-axis and not an average.

## Phase 2 — Compositional wordings

`attribute` is written in **6 wordings for the entire corpus**; `dispute` has 5,763
because it is built as a cross-product. Compose the rest the same way: opener ×
stance marker × rendering × closer. No model change and no run — this is authoring.

*Gate: ≥500 distinct outputs per move. Declare in advance that exact-match cell
accuracy falls as pools grow, or a success will read as a regression.*

## Phase 3 — Teach `cite`

The plumbing shipped in `crownless@4a24b6f6`: tomes have contents drawn from the
world, reading is a hash-neutral query, `# read:` lines reach the prompt, and `cite`
is the thirteenth move. Nothing has taught it.

Cite rows quote a read passage with the passage as a copy span — using the capability
that scores 60/60 rather than the one that scores 44/48. This is the first item on
this plan that produces sentences no authored pool ever contained.

*Gate: copy retention on cite = 100%, cite move accuracy ≥ 0.8, and a new metric —*
***novel sentence rate***, *the share of replies appearing in no authored pool, which
should exceed zero for the first time and must be entirely attributable to copied spans.*

**Fix first:** the vocabulary gate refuses words with no authored source, so book text
will trip it and every cite will score degenerate. The gate must accept a tome as an
authored source before this phase can be measured at all.

## Phase 4 — Possible worlds

Seeds are free and deterministic. Score worlds by notable events (`NotableStory` in
`cc_sim.c` is a first draft), sample conversations near the drama, and hunt
contradictory accounts of one event, which is the richest `dispute` material the sim
produces. The corpus currently balances *by event kind*, which treats a slain dragon
as interchangeable with a bakery's output.

This is the only phase that makes the *situations* interesting rather than the wordings.

## Phase 5 — The literature bridge, gated

Seed shelves from `zero-narrative` (Gutenberg Canada) re-skinned into world names.

One rule, non-negotiable, and it is the filter [Shumailov et al., Nature 631, 755–759
(2024)](https://www.nature.com/articles/s41586-024-07566-y) says recursion requires:
**a generation enters the corpus only if every word traces to an authored source or a
copied span.** Books make that rule generous rather than restrictive — each new book
widens the authored vocabulary legitimately.

## What this plan refuses

- **Scaling the model.** Nothing measured was a capacity limit; 5M learned twelve
  moves in 1.3 epochs once the anchor stopped fighting them.
- **Training an order of magnitude longer.** Ceiling at step 6,000; validation turns
  at 3.7 epochs.
- **Pretraining on 100M Wikipedia.** It dilutes a specialist to fix a fluency problem
  this model does not have (degenerate 0, leaked 0).
- **Self-training without the provenance filter.** The tails go first, and the tails
  are the sentences worth having.

## The method, which is the actual finding

Three failures in one day — a runtime cued with strings its weights had never seen, a
gate threshold that was an argparse default, and a distillation anchored on a premise
that was flatly false — were the same failure: something asserted about the model and
never measured. Each took under an hour to check. One was worth 28 guard rows.

**Every premise in a comment carries a number. Every run declares its gate before it
starts.**
