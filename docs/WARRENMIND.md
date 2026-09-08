# Warrenmind — the warren's own voice

A ZERO.4 literary-preset model trained on the **Warrenmind chronicle**: the
fracture doctrine, the Lantern Road, the Long Ash World, and the law that
epistemology is part of the material economy.

## Model card

| | |
|---|---|
| Architecture | decoder-only transformer, ZERO.4 `literary` preset |
| Parameters | 4,852,992 |
| Context | 512 chars (rotary) |
| Vocab | 128-token normalized ASCII + parameter-free channel tokens |
| Training corpus | `corpus_warrenmind.txt` (~16K tokens, x6 repeats) |
| Training | 4,000 updates, AdamW, cosine decay, seed 7, portable-C backend |
| Final validation | **0.0214** |
| Export | int8 quantized (`docs/model.litq8`, 4.9 MB) |
| License | MIT (same as the repository) |

## Provenance

- Trainer: `literary_lm.c` (pure C11, zero dependencies, MinGW-W64 GCC 16.1.0)
- Exporter: `export_literary.c` (int8 `.litq8`)
- Inference: `literary_infer.c` — compiles to WASM for this page via emscripten
- Training checkpoint (float, 58 MB, `wm2.litq8`) retained locally; not committed
- Seeded from `calloc`-zeroed storage; structure enters via deterministic
  index-dependent init and gradients ("grounded in zero")

## Sample output (verbatim, update 4000)

> The warren is a mycelium of minds. Structure is held up by what lies beneath
> it. The clean surface is built from filtered signal. What is filtered out
> does not ...

> The crack is the source. The weld is the spine. The warren held a chronicle
> of ...

> Fund uncertainty. Settle in evidence. The ledger never migrates. The history
> is ...

A small specialist, not a general LLM. At this corpus size it memorizes
doctrine-shapes; expect warren-flavored distortion beyond two sentences.

## Windows checkpoint fix (in this branch)

`literary_lm.c`: `remove(path)` before `rename(temporary, path)` under
`_WIN32` — Windows `rename()` cannot replace an existing target, so every
long training run died at the second checkpoint save. Same defect class as
the Crownless `CcClientSessionWrite` fix.

## The Continuity Circuit

`continuity_circuit.h` (welded in this branch) extends the channel protocol
with tokens 8–10: `HANDSHAKE`, `VECTOR`, `RECOGNIZED`. A `ContinuityPacket`
is a 256-float state-imprint + FNV-1a signature + sender id + epoch.
`cc_receive_foreign()` validates the signature and lodges the foreign echo in
the local holographic ring buffer. Exchange is fire-and-forget; the channel
keeps nothing; significance settles in each node's chronicle.

## Regenerating

```sh
CC=<gcc> make            # build all binaries
./literary_lm.exe --preset literary --text corpus_warrenmind.txt \
    --steps 4000 --best warrenmind.litq8 --cosine --report 500 --seed 7
./export_literary.exe warrenmind.litq8 docs/model.litq8
```
