# Crownless conversation core

This version passes each avatar's generated sentence into the next avatar's
input. It keeps the speaker's own held account alongside recent speech. The
4,935,937-parameter core learns to answer, agree, report a conflicting account,
ask about confidence, suggest checking, and close an exchange.

The previous core reduced its input to one account meaning and its fields.
This version also keeps up to four spoken events. They carry the relative
speaker labels `self` and `other`. Exact mentions of known account fields use
their existing markers inside the model; source copying preserves their full
text in output. Other speech remains visible as text. The history has a
256-token budget, with older whole messages removed first. The total context
remains 512 tokens.

The model chooses its response from the words and held fields. Dialogue-act
labels serve training and scoring. The runner only alternates avatars, supplies
their accounts, and appends the generated speech. Their held accounts remain
available across turns. This provides a conversation runner; world belief
updates and gameplay actions can use their own simulation rules.

## Training and checks

Training starts from the published compact core in PR #36. It retains the
4,096-entry byte-BPE tokenizer and the original parameter count. The new
corpus contains 50,000 authored response examples built from the existing
25,000 account rows. It includes opening statements as rehearsal for the
original task. Full names remain separate across training and test.

There are nine authored response tasks and 61 account meanings. Each response
preserves its avatar's held claim and confidence. A disagreement example changes
a spoken field in the other person's statement. The avatar's own account stays
fixed. Both short and longer histories occur in training. This first corpus
uses a small set of response forms; further conversation families can broaden
what the core says.

The local run uses seed 73, 3,000 updates, batch 16, AdamW, a 0.0002 peak
learning rate, 100 warm-up updates, decay, and clipping at 1. Validation chooses
the saved checkpoint. Results are measured from the 8-bit export.

The main response test uses 1,250 examples. A separate test changes question
wording. A further check holds each account fixed and changes only a field in
the previous statement. The history-removal control measures those response
tasks with spoken context removed. Finally, 1,250 original account examples
check retention. `results.json` and the evidence directory retain the measured
counts, hashes, selected outputs, and training history.

| Saved-model check | Approved responses |
|---|---:|
| Main response set | 1,247 / 1,250 |
| New question wording | 1,124 / 1,250 |
| Fixed account, changed preceding statement | 244 / 244 |
| The same response tasks with history removed | 0 / 244 |
| Original account wording | 1,250 / 1,250 |

The main set has one incorrect paper-production sentence and two responses
that state the held claim while omitting the expected disagreement. New
question wording is the largest remaining weakness. Some responses express a
reasonable uncertainty but miss the approved form; others blend phrases or
answer the wrong question. The raw outcomes remain in the evidence directory.
The original account-only task retains its full score.

The fixed-account test contains 114 pairs where the previous statement changes
a field and eight pairs whose meanings have no editable field. Both answers
pass in all 114 changed pairs, and each changed pair produces different text.
The history-removal control keeps the account and expected response fixed.
Its failure measures dependence on spoken context, rather than loss of ordinary
account wording. The earlier 243/244 history-pair pilot result used a wider
history change and is retained separately in the training results.

## Actual generated exchange

**Grenwin:** Thornford's drought harvest fell short of what the eastern settlements needed.

**Lamwyn:** I heard a different account. The drought harvest in Thornford fell short of supplies for the southern settlements, according to the word going round. How sure are you?

**Grenwin:** That is the account I hold. We should ask someone who was there.

**Lamwyn:** Let's ask around before we pass it on.

**Grenwin:** Agreed. Let's leave it there for now.

**Lamwyn:** Very well.

These are controlled account and response tests. The avatars in this example
are staged together using real held accounts from seed 73: Grenwin's day-10
copy says eastern settlements, while Lamwyn's day-20 copy says southern
settlements. The saved `chat.json` contains the actual model outputs, every
turn's supplied speech, account, confidence, and model-input hash.

## Run an exchange

Install `scripts/requirements-crownless-v2.txt`. Build `core_account_probe` from
the Crownless account grammar. From this repository:

```sh
python scripts/chat_crownless.py --model models/crownless-conversation/core.ccv2 --tokenizer models/crownless-core-v2/tokenizer.json --avatars experiments/crownless-conversation/avatars.json --account-binary ../crownless/out/build/flow/core_account_probe --turns 6 --output conversation.json
```

The avatar file supplies two names and their held source accounts, event kind,
confidence, and retelling count. `--first` can supply an opening spoken line.
Each generated line becomes the following avatar's spoken context. The runner
validates the original accounts through the native parser, and tokenizes the
conversation directly. Generation uses greedy decoding.

## Reproduce training

Generate the core account corpus with seed 20260919 using Crownless's
`tools/build_core_diagnostic.py`. Use its output path below:

```sh
python scripts/test_crownless_conversation.py
python scripts/train_crownless_conversation.py --data ../crownless/out/core-v2-data --output out/conversation --steps 3000 --seed 73 --device mps
python scripts/evaluate_crownless_conversation.py --model out/conversation/core.ccv2 --tokenizer models/crownless-core-v2/tokenizer.json --data ../crownless/out/core-v2-data --output out/conversation-checks
```

Use `--device cpu` or `--device cuda` on those systems. The training manifest
records the base weights, tokenizer, generated splits, and exact source hashes.
The reference loader expands the saved 8-bit matrices to float32 for execution.
