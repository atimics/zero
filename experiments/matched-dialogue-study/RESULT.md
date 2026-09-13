# Matched inputs help, with a coverage cost

The broader run produces useful conversation on several new accounts. It also loses reliable speech on six event types absent from this dialogue set. This is a partial research result.

Both avatars hold the same account in training and evaluation. Each model starts from the shipped 4,935,937-parameter core and runs 1,200 CPU steps with identical batch size, learning rate, seed, and replay fraction. The small arm uses twelve approved conversations; the broader arm uses 46 usable conversations, including 34 assistant-written candidates. Two raid examples have unsupported source packets and are excluded. The original 48-draft batch remains unchanged.

Authored openings and the shipped model's own openings provide two history variants where the wording differs. Partial person names become their full source names so their mentions use existing copy slots. This yields 80 dialogue rows in the small arm and 296 in the broader arm. These are variants of twelve and 46 conversations, respectively.

## Generated conversation review

An assistant read all sixteen four-turn conversations from each arm:

| Outcome | Twelve examples | Broader set |
| --- | ---: | ---: |
| Natural, relevant exchange | 4 | 6 |
| Relevant but repetitive | 2 | 4 |
| Broken language or facts | 10 | 6 |

These are qualitative assistant labels, with case IDs and definitions in `review.json`. They describe this sample; human preference remains separate. The baseline uses its fixed agreement/checking sequence. All baseline and candidate outputs are preserved in [SAMPLES.md](SAMPLES.md).

The broader model says:

> Quillen Longreckon joined The Ditch Parliament after seeking food at Silverwick.
>
> For long?
>
> Days.
>
> A long wait for a meal.

On a fresh calving account:

> Alderwatch raised calves and added older calves to the working herd.
>
> The new one is still growing?
>
> Yes. The older one joined the workers.
>
> That is good news for the herd.

The goblin conversation retains the new rank Voice, and a death conversation asks where Skella Mailmender lived and answers Alderwatch. The twelve-example arm also succeeds on some of these cases. Wider coverage adds useful responses about cattle loss, lambing, and flock culling, though some endings repeat.

## Regression and limits

All six broad-arm failures concern event types absent from the new dialogue set: route closure, food shortage, treasure making, woodcutting, quarry work, and shearing. For example, the quarry account produces “Silverwick made bread from Silverwick.” The empty food store produces “Gloamgate has made bread.” These change the event materially.

The opening replay rows cover the same limited sources as the dialogue training. This pattern points toward a retention problem during fine-tuning. A follow-up should retain opening rehearsal across the full original event grammar and add broader conversations before choosing a game model. This study alone does not separate the effects of coverage, wording variation, name normalization, and the new matched input from all other training choices. The two arms within this study share those choices, so their direct difference is the set of examples.

## Measurements and data separation

All 64 lines from each arm reached EOS. The broad arm has 57 distinct lines and one line with a replacement character. The small arm has 61 distinct lines. Higher diversity alone did not track the qualitative outcome.

On the fixed every-other-row training check, the small arm exactly reproduces 29/40 targets and the broader arm 106/148. These are recall diagnostics; alternative valid wording can differ. Training took 80.7 and 74.6 seconds, respectively. Both candidates were reloaded from their int8 exports for scoring.

Worlds 1601 and 1602 were freshly simulated. Sixteen accounts were fixed before baseline generation. Test account text is separate from all usable training sources after case folding and numeral normalization. Nine opening inputs share an existing field-slot pattern, while none matches both that pattern and its literal copy values. Thus part of this test measures transfer to new people or places within a learned event pattern. Shared grammar and possible pretraining exposure remain.

The checkpoints are fixed final-step results. The outputs were inspected after each arm completed; the second arm's training configuration stayed fixed. This work leaves the game on its existing model.

## Reproduce

From the repository, using PyTorch 2.8.0, NumPy, and tokenizers:

```sh
python scripts/run_matched_dialogue_study.py --output /tmp/crownless-matched-new
python scripts/audit_matched_dialogue_inputs.py /tmp/crownless-matched-new
python tests/test_matched_dialogue_study.py
```

The input and model hashes are recorded in `evidence/manifest.json`. Candidate weights remain local at `/private/tmp/crownless-matched-study-run/small.ccv2` and `/private/tmp/crownless-matched-study-run/broad.ccv2`. Their hashes are in `evidence/results.json`.

Four matched-study tests, four earlier pilot tests, and four existing conversation tests passed. They cover both-speaker account inputs, relative history, opening variants, full-name normalization, copying, and split checks. This study depends on the reviewed-dialogue pilot in PR #42.
