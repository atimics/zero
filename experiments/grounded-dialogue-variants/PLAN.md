# Grounded dialogue: question-variant follow-up

Status: results recorded · 2026-09-21 · successor to `experiments/grounded-dialogue/`

## The problem, measured

The first grounded-dialogue bank gave every meaning exactly one question and one
answer. The model learned a usual answer per event and ignored the question.
On eight held-out questions that asked for a different fact of the same event,
both arms failed every case:

| Check (prior arm) | Result |
| --- | ---: |
| In-bank answers | 1/129 |
| Exact copy actions on field-bearing answers | 9/99 |
| Reworded original-fact probes | 0/8 |
| Different-fact contrasts | 0/8 |

The editorial review concluded that the exchange was a quiz about the first
sentence and that the response targets needed revision before more training.

## The intervention

Keep the original exchange and add one **alternate-fact** question per meaning
that asks about a different copyable account field. The answer binds that field,
so the question, not the event, selects what is copied. The alternate wording is
deliberately different from the held-out `question-contrasts.json` probes, so
the follow-up measures question routing rather than memorised wording.

Nothing else changes: same base checkpoint (`244a809a…`), same seed 915, same
2,400 steps, same 12-dialogue/4-rehearsal batch, same learning rate, same fixed
final checkpoint, same metric. `scripts/build_grounded_bank_v2.py` rebuilds
`input/dialogue-bank-v2.json`; `scripts/grounded_dialogue.py` now accepts one
exchange or a list of exchanges per meaning.

## Gates, declared before the run

| Metric | Gate |
| --- | --- |
| Different-fact contrasts (exact) | **> 0**, target ≥ 6/8 |
| Reworded original-fact probes (exact) | ≥ 6/8 |
| In-bank answers | **not below 129/129** |
| Copy actions | **99/99** |
| Opening forms retained | 61/61 |
| Degenerate / leaked lines | 0 / 0 |

## Files

- `experiments/grounded-dialogue/input/dialogue-bank-v2.json` — the variant bank
- `scripts/build_grounded_bank_v2.py` — its builder
- `scripts/grounded_dialogue.py` — variant-aware compiler
- `scripts/run_grounded_dialogue.py` — the comparison
- `scripts/check_grounded_questions.py` — the held-out probes
- `tests/test_grounded_variants.py` — coverage, encoding, field, and wording checks
