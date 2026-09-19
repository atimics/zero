# Pilot sample findings — 19 September 2026

All 800 registered continuations were generated from the four selected seed-7
checkpoints. Both comparisons passed the repeated four-gram limit. A/B passed
the 2% loss threshold; B/C achieved 0.574% and falls short of that threshold.

Codex read 20 pairs per comparison, using case IDs 1, 11, 21, through 191.
The selection was saved before reading. All 40 judgments were saved before
opening the answer keys. Both comparisons use the same 20 prompts, so this
review covers 80 continuations on 20 distinct prompts.

| Comparison | Candidate wins | Control wins | Skips | Loss improvement |
| --- | ---: | ---: | ---: | ---: |
| B over A, 27,157,191 presentations each | 6 | 2 | 12 | 2.716% |
| C over B, 100,499,909 presentations each | 7 | 3 | 10 | 0.574% |

Preferences mostly reflect clearer local wording. All 80 reviewed
continuations struggled to sustain a clear scene consistent with the prompt.
Common problems were changing characters, confused family relationships,
broken event order, invented words, and dialogue that lost its subject.
The large skip counts reflect ties or unclear preferences.

In B/C case 101, C retained Janet and the doctor longer, but later called her
Mr. Janet. In case 41, B described a father as a poor girl. In case 181, C
mentioned the requested mirror but lost its purpose, while B changed two men
into wagons. These examples show why a relative preference can coexist with
weak scene quality.

| Arm | Mean repeated four-gram excess |
| --- | ---: |
| A in A/B | 0.0639% |
| B in A/B | 0.2138% |
| B in B/C | 0.2994% |
| C in B/C | 0.3118% |

Each pair stays within the allowed increase of one percentage point.
Exact phrase repetition is a narrow measure; the reading also found repeated
vague ideas and sentence patterns. Rates were recomputed from every generated
continuation and matched the worker's saved metrics.

The designated 200-pair human review remains pending. These automated sample
judgments are descriptive and separate from the human vote records. A/B's
overall decision remains open; B/C falls short on the loss gate. The next
registered step is the full A/B human review. A separate development experiment
should target coherent paragraphs and stable characters, with model capacity
and training design as candidate causes to investigate.

Scope: one training seed, one automated reviewer, systematic prompt selection,
and the protected A validation set. Several selected prompts are essay or
descriptive passages. The findings apply to these prompts and checkpoints.

## Evidence identities

Runtime evidence is retained outside Git in
`output/zero-oregon-samples-20260919` under the ilxyr workspace. It contains
the full packets and HTML pages, individual judgments with reasons, the fixed
review plan, the pre-unblinding receipt, the report, and cloud receipts.
Sampling used the frozen 128-token prompts, 256 new tokens, temperature 0.7,
top-k 40, repetition penalty 1.1 and seed 107 on an Oregon A10G with BF16.
Additional training presentations: zero.

| Artifact | SHA-256 |
| --- | --- |
| Cloud manifest | `6a337f64089d69d5c72792386b5fa02b801101846f0b515bee4b97ed9bcc1618` |
| A/B packet | `ad5a70a3c23265d175ae1028c4891ad35939b0f01847eaf6a54709833d9a624d` |
| B/C packet | `e3ff54905fd7b97caf3f02208f0a60d7bd232f91de3999b8e31e573a05ccbe07` |
| Review plan | `d811d29d5d6e2450ae66022f52a89d55c34f8ac0322de28008865ae90722f917` |
| Forty blind judgments | `1831b0c93e1d49ad52eb768822a22a09045a4d4ef7be437dd762f2b47640895b` |
