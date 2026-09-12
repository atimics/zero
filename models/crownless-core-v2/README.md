# Crownless core 5M v2

This experimental core turns a selected Crownless account meaning into a short
English sentence. It has **4,935,937 parameters**. `core.ccv2` stores row-wise
8-bit matrices and float scales in **5,044,760 bytes**. The tokenizer is lossless
byte BPE with 4,096 entries. The weights use the [MIT license](../LICENSE).

The game parser supplies the meaning, spoken field roles, and knowledge labels.
The model learns the sentence and chooses which source fields to copy. Complete
names stay in a separate source buffer. The external input is a short event
list with its selected account last. In this version the encoder uses that
selected account for realization.

The model was trained from fresh weights on 25,000 paired examples from the
shared game grammar: 61 patterns across 44 event kinds. Confidence cues control
uncertain and widely retold wording. Voices, jobs, and broader English training
are later experiments. [Training and evaluation report](../../experiments/crownless-core-v2/README.md).

## Speak from an account

Install `scripts/requirements-crownless-v2.txt` and build `core_account_probe`
from [Crownless PR 689](https://github.com/atimics/crownlesscarriage/pull/689).
Run from the ZERO repository, replacing the binary path with your build:

```sh
python scripts/speak_crownless_v2.py --model models/crownless-core-v2/core.ccv2 --tokenizer models/crownless-core-v2/tokenizer.json --account-binary ../crownless/out/build/core-v2/core_account_probe --kind 133 --event 'Mara Venn posts a notice at Thornford: Relief charter.'
```

The runner checks both tokenizer and grammar hashes. The Python reference
loader expands the compact weights to float32 for inference. Native neural
inference is a later integration step; the shared grammar already runs in the
game's C speech path.

Weight SHA-256:
`fdc469199c10ed24795b8f66248738c24b9f3bf4d12c52342d7f588ae95f2c02`
