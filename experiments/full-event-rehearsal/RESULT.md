# Full-event rehearsal result

Keeping the original event coverage in rehearsal helped the 4.94M model retain its old topics while learning short dialogue. The next data expansion should give those topics their own natural exchanges, with explicit copy targets for every named fact.

## Controlled comparison

Both runs used the same base, seed, 296 dialogue rows, and 1,200 training steps. Each batch held twelve dialogue rows and four rehearsal rows. Narrow rehearsal used 46 openings. Full rehearsal used 3,660 openings across all 61 meanings plus 2,640 old conversation rows on meanings outside the new dialogue coverage. This measures the combined effect of broader event coverage and old response rehearsal.

| Measure | Base | Narrow | Full |
|---|---:|---:|---:|
| Original accepted opening forms | 61/61 | 43/61 | 55/61 |
| Completed world-test lines | 64/64 | 64/64 | 64/64 |
| Distinct world-test lines | 21 | 61 | 45 |

The opening check accepts the original grammar forms. Natural paraphrases and facts spread across turns can also be valid. Distinct lines measure variety; broken text can also be distinct.

The sixteen world cases came from fresh simulation seeds 1801 and 1802. An assistant read the complete exchanges. The full arm kept seven natural exchanges, restored seven stock exchanges on previously missing topics, repeated a reply in the flock case, and produced a broken final line in the cattle case. This small review is descriptive. Human preference remains to be measured.

For example, the full model generated:

> A: Gloamgate made bread from wheat.  
> B: From wheat?  
> A: Yes.  
> B: I wonder what the bread is like.

The full model's paper exchange introduced paper first, then supplied the material when asked: “From what?” / “Rags.” Reading the full exchange revealed facts that the opening-form score missed.

## Direction-copy follow-up

Inspection found a real grounding error: a harvest reply said “Those to the south” when the account named Yorashormere. The authored training phrase paraphrased “southern settlements,” which prevented the existing exact-field matcher from making a copy target.

An exploratory full-arm rerun normalized directional place phrases to their account field before training. It used the same budget and reached 55/61 opening forms. The harvest reply became “The Yorashormere, from what I heard.” This fixes the destination in that observed case, with an awkward article remaining.

Other outputs changed too. The bandit opening substituted “the Unpaid Company” for the account's “The Ditch Parliament,” and the bakery closing became broken. Cattle and flock replies repeated. These regressions make the follow-up a diagnostic candidate. Further work should audit copy targets across all entity phrases and test substitutions across fresh names and places.

## Evidence and scope

The input directory contains rule-generated rehearsal, simulation accounts, and generation receipts. The evidence directories contain manifests, training logs, every generated test conversation, opening checks, and source hashes. The original runner is saved under `evidence/initial/source-runner.py`; the current runner adds optional direction normalization. Twelve of the 48 authored scenes had human approval; 46 scenes fit the packet grammar and were used here.

Training and evaluation worlds differ, and the test accounts differ after case folding and number normalization. Grammar patterns remain shared. The 61-case diagnostic uses separate generated names. These checks measure continuation and retention within the original grammar.

All candidate results use reloaded int8 files through the checked Python loader. Native game integration needs model-bound tables and a separate parity check. The deployed model stays at the existing shipped version. Candidate files remain in the local run directories recorded below; their hashes appear in each results file.

## Reproduce

Use Python 3.11, Torch 2.8.0, and the project's tokenizer dependencies from the repository root:

```sh
python scripts/run_full_event_rehearsal.py --output /tmp/rehearsal-run
python scripts/check_full_event_retention.py /tmp/rehearsal-run
python scripts/check_rehearsal_exchanges.py /tmp/rehearsal-run
python scripts/run_full_event_rehearsal.py --output /tmp/rehearsal-directions --arms full --copy-directions
python scripts/check_full_event_retention.py /tmp/rehearsal-directions --arms base full
python scripts/check_rehearsal_exchanges.py /tmp/rehearsal-directions
```

Local runs: `/private/tmp/crownless-full-event-rehearsal-run` and `/private/tmp/crownless-full-event-rehearsal-direction-run`. Each run has its exported candidate files. Seven new regression tests, eight earlier dialogue tests, and four conversation tests passed locally. The new tests are also wired into the Torch CI job.
