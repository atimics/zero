# Grounded dialogue: editorial review

Keep the explicit field-copy compiler and the wider event coverage. Revise the conversation targets before further training. The expanded model gives clear, grounded lines, but the exchanges often feel like a quiz about the first sentence.

## What the conversations show

A speaker gives several facts. The next speaker asks for one of those facts. The answer repeats it. A final line adds a general hope or wish. This pattern is visible across routes, reserves, treasure, quarrying, and shearing. The wording changes while the exchange stays much the same.

The death case makes the tradeoff especially clear. The earlier model says “Cinda Orehauler has died.” Asking where Cinda lived then adds something. The expanded model gives Silverwick in the opening and immediately asks where Cinda lived. The data made the exchange less natural despite preserving the facts.

The expanded flock-cull exchange works better. Its opening mentions slaughter and the autumn cull; the question asks where the mutton went. “Into the local store” adds a fact that the listener has yet to hear. That is a useful pattern for the next draft.

My review of all sixteen pairs preferred the earlier version in six cases and the expanded version in four. Six pairs need a rewrite on both sides. These are assistant editorial judgements. Full reasons and every generated line are in `evidence/assistant-review.json` and `evidence/SAMPLES.md`.

## Measured result

Both models started from the same shipped 4.94M base and trained for 2,400 steps with seed 915. The earlier-data arm used 296 dialogue rows and 6,300 rehearsal rows. The expanded arm used 14,640 dialogue rows and 3,660 opening rows across all 61 supported meanings. Each batch held twelve dialogue rows and four rehearsal rows. CPU training took about three minutes per arm.

| Check | Earlier data | Expanded data |
|---|---:|---:|
| Original opening forms retained | 58/61 | 61/61 |
| Exact answers with bank questions | 1/129 | 129/129 |
| Exact copy actions on field-bearing answers | 9/99 | 99/99 |
| Completed fresh-world lines | 64/64 | 64/64 |
| Exact answers to eight reworded questions | 0/8 | 8/8 |
| Correct copy actions when eight questions ask for a different fact | 0/8 | 0/8 |

The last check is decisive for interpretation. For a bandit event, asking where the person sought food still produces the group they joined. For a treasure event, asking where it was made still produces the object. The model appears to have learned a usual answer for each event and turn. The perfect copy score shows that it can fill that answer with a changed source name. Question choice remains a separate weakness.

The reworded-question check was written during training before expanded outputs were inspected. The different-fact check was added after inspection and is exploratory. Each source field substitution reused the same meaning and question structure. Shared templates make these narrow grounding checks. The sixteen free-running exchanges came from fresh simulation worlds 1901 and 1902, with four generated turns each.

## Next target design

Give the opening one useful fact. Let a question ask for information still absent from the spoken exchange. Let another speaker react when a question would add little. Use different questions and answers for the same event. Give exchanges room to end after two or three turns.

For example, this is an assistant-written revision:

> A: They’ve shut the bridge at Alderwatch.  
> B: What about the relief convoy?  
> A: Held up.  
> B: That’s a bad time to close a bridge.

The existing account supports the closure and delay. The last line expresses a reaction. Each turn has a purpose.

The proposed alternate-question training and fresh checks are saved in `proposals/`. They remain untrained. This editorial review takes priority over that follow-up because the opening-and-response structure needs revision too.

## Evidence and validation

The compiler creates explicit UTF-8 copy spans from numbered account fields, rejects unavailable spoken fields, and handles changed names and hidden identities. All examples passed the existing encoder's span and context checks. Complete generated test names are separate from training names. Event grammar remains shared.

The run manifest pins the base, tokenizer, code, and data hashes. Candidate files remain at `/private/tmp/crownless-grounded-dialogue-run/prior.ccv2` and `/private/tmp/crownless-grounded-dialogue-run/expanded.ccv2`; hashes are in the saved results. The initial runner source is preserved to match its manifest. Evaluation used reloaded int8 files through the checked Python loader. Game integration needs its own model-table and native parity work.

Eleven new compiler/review tests, fifteen earlier dialogue/rehearsal tests, four conversation tests, and five earlier A/B tests passed locally. The review page was checked in the browser for complete exchanges, progress, and undo. Its test votes use a separate local address and file. The user requested assistant review; the editorial decisions are recorded separately from human votes.

## Reproduce

From the repository root with Python 3.11, Torch 2.8.0, and the project's tokenizer dependencies:

```sh
python scripts/run_grounded_dialogue.py --output /tmp/crownless-grounded-run
python scripts/check_grounded_questions.py /tmp/crownless-grounded-run
python scripts/check_grounded_questions.py /tmp/crownless-grounded-run --contrasts
python scripts/build_grounded_review.py /tmp/crownless-grounded-run --output /tmp/crownless-grounded-review
```

The PR contains the experiment and its limitations. The existing game model remains the deployed version.
