# Question-variant follow-up: result

One alternate-fact question per meaning fixed the different-fact failure without
losing the in-bank scores. The question now selects the copied field instead of
the event.

## Measured result

Both arms start from the same shipped 4,935,937-parameter base and train for
2,400 steps with seed 915. The prior arm uses the original one-question bank
(296 dialogue rows). The expanded arm uses the variant bank (26,400 dialogue
rows) and the same full-event rehearsal. Each batch holds twelve dialogue rows
and four rehearsal rows. CPU training took 165.6 s (prior) and 147.5 s
(expanded).

| Check | Prior | Expanded |
| --- | ---: | ---: |
| In-bank answers, exact | 1/129 | **129/129** |
| Exact copy actions on field-bearing answers | 9/99 | **99/99** |
| Opening forms retained | 58/61 | **61/61** |
| Reworded original-fact probes, exact | 0/8 | **6/8** |
| Reworded original-fact probes, copies | 1/8 | **6/8** |
| **Different-fact contrasts, exact** | **0/8** | **6/8** |
| **Different-fact contrasts, copies** | **0/8** | **7/8** |
| Fresh-world lines completed | 64/64 | 64/64 |
| Distinct fresh-world lines | 48 | **64** |

The primary gate is met: different-fact exact answers rose from 0/8 to 6/8, and
correct copy actions to 7/8, while the in-bank answer and copy scores did not
fall. All 64 fresh-world lines stopped with no degenerate or leaked output.

## What still fails

Four of the sixteen held-out probes miss, and each is a routing or phrasing
miss rather than a copied-name error:

- `prophecy_delivered_0`, "What did the company deliver?" (object) answered
  `To Yorumowholt.` (place): the wrong field.
- `paper_milled_0`, "What was the paper made from?" (material) answered
  `In Kelilashmere.` (place): the wrong field.
- `harvest_failed_0`, "Where did the harvest fail?" answered `Kelilowden, I
  think.` — the right field, but the `In` prefix is dropped, so the text is not
  exact.
- `shortage_0`, "Where are people hungry?" answered `Into them.` — wrong.

Two questions still route to the place field when another field was asked. This
is the residual failure the typed fact-selection design (P1b) would address.

## Scope and limitations

- One training seed per arm; the comparison is conditional on these weights.
- The variant bank is assistant-authored and awaits human review.
- The probes are eight in-grammar cases with shared templates; they are not a
  broad generalization test.
- Copy scoring measures exact copy actions, not claim fidelity, roles,
  quantities, or source. Those need the separate checks in issue 809.
- The account opening still states all fields, so the exchange can still read as
  a quiz. This change fixes routing, not opening naturalness.

## Reproduction

From the repository root with Python 3.11, Torch 2.8.0, and the pinned
tokenizer dependencies:

```sh
python scripts/build_grounded_bank_v2.py --output \
  experiments/grounded-dialogue/input/dialogue-bank-v2.json
python scripts/run_grounded_dialogue.py --output /tmp/crownless-grounded-v2
python scripts/check_grounded_questions.py /tmp/crownless-grounded-v2
python scripts/check_grounded_questions.py /tmp/crownless-grounded-v2 --contrasts
```

The candidate files stay in the local run directory; their hashes are in
`evidence/results.json`. Evidence in `evidence/` keeps the manifest, both
candidate diagnostics, both fresh-world rollouts, the two probe results, and
selected samples.

## Next step

The two residual routing misses motivate P1b: typed fact selection, where the
model chooses a fact reference and the renderer owns the wording. That is the
same mechanism that fixed stance (8% → 61% intervention) and is already
described in `tools/dialogue/KNOWLEDGE-COVERAGE.md`.
