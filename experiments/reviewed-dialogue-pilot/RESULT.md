# Reviewed dialogue pilot: useful fit, failed transfer

The 4,935,937-parameter shipped model was tuned locally on the twelve conversations marked Keep in the fresh review. Those conversations supply 42 next-line targets. Each batch contains six dialogue rows and two original grounded opening rows. The tokenizer and model architecture stay fixed.

A fixed 200-step CPU run was followed by a 1,000-step fit diagnostic after poor generated conversations were inspected. Each run started from the same shipped model, seed 912, AdamW learning rate 0.00005, batch size eight. The first 200 steps reproduce the same losses and candidate hash across the original run and its diagnostic repeat. The 1,000-step result is exploratory because the evaluation outputs had already been inspected. Final steps determine the checkpoints.

## Results

| Check | 200 steps | 1,000 steps |
| --- | ---: | ---: |
| Exact approved next lines with intended history | 16 / 42 | 34 / 42 |
| Generated test lines reaching EOS | 48 / 48 | 48 / 48 |
| Distinct test lines | 38 / 48 | 44 / 48 |
| First test lines identical to shipped model | 12 / 12 | 10 / 12 |
| Training time in recorded diagnostic run | 14.7 seconds | 43.9 seconds |

Exact match measures reproduction of the targets. Alternative valid phrasing is possible. Line diversity and termination are diagnostics; generated text remains the quality evidence. Assistant inspection of the twelve conversations at each checkpoint found broken language, misplaced responses, and factual problems. These candidates remain research artifacts.

The 1,000-step candidate produced this conversation on an account of a goblin becoming Voice:

> A goblin rose to Voice in the dragon cult through service.
>
> Keeper?
>
> Yes. That is the title.
>
> Good for them.

The rank changes within the exchange. Keeper appears in the training example. Another harvest account receives “I wonder what the new calf looks like.” These are concrete signs of responses carrying across the wrong situations. An uncertain bakery account becomes “Gloamgate has made bread,” losing its uncertainty. A harvest line becomes “Thornford’s drought harvest fell short, and the story is right,” dropping the destination and changing the uncertainty signal.

## Input boundary

The authored review assumes the first speaker holds the account and the listener learns from speech. The pilot models that explicitly: holder turns include the account fields and meaning ID; listener turns have only the previous speech. The game usually provides an account to both avatars. The shipped baseline behaves poorly on a history-only listener, so that comparison combines writing adaptation with a new input condition.

A second diagnostic supplies the account to both avatars during evaluation. The shipped model repeats its familiar agreement/checking sequence. The tuned candidates also produce broken language in this condition. Those outputs are saved separately as `before-shared.json` and `after-shared.json`. The input mismatch and tiny training set limit causal claims. A follow-up should train and evaluate matched game conditions with broader dialogue examples and separate speaker knowledge.

## Split and evidence

Training worlds are 1201, 1202, and 1203. Evaluation world 1401 was freshly simulated for 365 days and finished valid. Twelve evaluation accounts were selected across twelve event families. Exact account overlap is excluded after case folding and numeral normalization. Shared names, grammar, semantic patterns, and possible pretraining exposure remain. These are held out from this fine-tune, rather than a claim of wholly novel language.

The original 200-step plan used fixed final-step selection. Extra diagnostics and the longer fit check were added after inspecting the initial outputs. Both checkpoints and all evaluation text are retained. No human preference result is claimed for model generations.

`input` preserves the authored drafts, their source rows, and the fresh evaluation source receipt. `evidence/steps-200` and `evidence/steps-1000` contain training rows, losses, manifests, target-fit outputs, and the two evaluation conditions. The output manifest lists exact model hashes.

The native game probe rejected the new artifact because its loader checks the shipped file's exact hash. Inspection of `CcCoreModelLoad` confirmed that binding. Python's checked int8 loader ran both candidate evaluations. Native candidate parity remains a later integration check after rebuilding the model-bound tables.

## Reproduce

Use the environment recorded in the manifests (Python with PyTorch 2.8.0, NumPy, and tokenizers). From this repository:

```sh
python scripts/run_reviewed_dialogue_pilot.py \
  --input experiments/reviewed-dialogue-pilot/input \
  --eval experiments/reviewed-dialogue-pilot/input/evaluation-accounts.jsonl \
  --base models/crownless-conversation/core.ccv2 \
  --tokenizer models/crownless-core-v2/tokenizer.json \
  --steps 200 --output /tmp/crownless-pilot-new
```

Use `--steps 1000` and a separate output directory for the fit diagnostic. The runner checks the shipped model hash. Each local output contains its quantized candidate. Candidate files from these runs are retained at `/private/tmp/crownless-reviewed-pilot-diagnostic/candidate.ccv2` and `/private/tmp/crownless-reviewed-pilot-1000/candidate.ccv2`.

Four focused tests passed for listener knowledge isolation, byte-accurate copy positions, whole-name copying, and the numeral-normalized split key. All 42 targets round-trip through the pinned tokenizer. Source and evidence hashes are saved in `receipt.json`.
