# Crownless core: 4.85M parameters

This builds a core speech model from ZERO's Gutenberg checkpoint, the Crownless gossip rules, and varied event facts. Model input is a short list of plain event lines. The next line is the NPC's speech.

The final dataset contains 32,000 training rows: 24,000 paired fact variations and two copies of the 4,000 original game examples. It covers twelve gossip topics. Names include the game's naming styles, long names, titles, factions beginning with "The", and names built from many short sound combinations. Training tunes all core weights.

Each pair changes one named fact in the final event while keeping the earlier events fixed. A second test uses reserved input sentence forms. The strict test filter removes a whole pair if any topic name appears anywhere in either stage's training text, including inside another name. This leaves 216 examples for name copying and 206 for new wording. Both retain complete pairs. The original 500 game examples and sixteen editorial examples are known regression sets, since their outputs were inspected during this work.

The automatic score checks exact reference-name presence and whether both members of a pair follow the changed name. This is one part of factual accuracy. A valid "someone" sentence can omit a reference name under uncertainty. Exact reference sentences are also counted; alternate valid wording can lower that score.

The first paired-data stage improved reserved-name copying but failed on the original game names. Its complete evidence remains in `paired-stage/`. The final stage adds original game replay and wider name forms. Training settings also changed across stages, so these comparisons measure the whole training recipe.

The run used an Apple M4 Max locally. The selected weights were chosen by ordinary validation loss on 240 examples. Some validation names occur as parts of other training names; the separate strict test results exclude those overlaps. The original editorial challenge includes war and dragon-brood events, which are outside the twelve training topics.

## Results

| Check: every reference name present | First pilot | Paired stage | Final mixed stage |
| --- | ---: | ---: | ---: |
| Original game events | 8 / 500 | 0 / 500 | **435 / 500 (87.0%)** |
| Strict new names | 0 / 216 | 5 / 216 | **211 / 216 (97.7%)** |
| Strict new wording | 0 / 206 | 8 / 206 | **100 / 206 (48.5%)** |

The final model followed the changed name in both members of 105 / 108 strict pairs. Exact reference sentences matched on 261 / 500 original game examples, 96 / 216 strict name examples, and 32 / 206 strict wording examples. It ended all original-game and strict-name sentences, and 204 / 206 strict-wording sentences, within the 160-character limit.

Remaining errors include substituted names and wrong event types under new wording. All five strict name misses contain a changed or shortened name. On manual review, five of the sixteen known editorial examples preserve their main facts with clean names (cases 4, 5, 8, 11, and 13). These results make the checkpoint an experimental core model.

The final pass selected update 3,500 from 4,000 updates. Its ordinary validation loss is 0.01028 and ordinary test loss is 0.01236 per output character. The two local passes took 703.9 and 792.4 seconds including their built-in evaluation, about 25 minutes combined. The broader generation comparisons ran afterward on the CPU with cached context.

Actual output:

> Forgen Miller posted something in Thornford.

This followed `? Forgen Miller posts a notice at Thornford: Relief charter.` A retold, uncertain account of The Ditch Parliament produced:

> Displaced workers gave The Ditch Parliament support on the old road, if the story is right.

The native 4,920,400-byte export also produced that second sentence in its smoke check. The weights are at [models/crownless-core-5m-v1.litq8](../../models/crownless-core-5m-v1.litq8). The floating-point checkpoint and complete corpus are saved in the local run directory recorded in the metadata.

## Reproduce

Use Python 3.11, `torch==2.8.0`, and `numpy==2.2.6`. Supply the Crownless v2 corpus from public commit `79706fad39e95d8f20e763e277a42efb7b24896f` in `atimics/crownlesscarriage`. The initial Gutenberg checkpoint hash is `9ea700a76eed50eb2f13e57cd00567e80a7e43449a562829ba74d35363021244`.

```sh
python scripts/build_crownless_facts.py --source CORE_V2 --output PAIRED
python scripts/tune_crownless.py --base GUTENBERG/best.ckpt --data PAIRED \
  --output PAIRED_RUN --device mps --steps 6000 --batch-size 16 \
  --eval-every 500 --lr 0.0002 --max-seconds 900
python scripts/build_crownless_mixed.py --source CORE_V2 --output MIXED \
  --prior-data PAIRED
python scripts/tune_crownless.py --base PAIRED_RUN/best.ckpt --data MIXED \
  --output MIXED_RUN --device mps --steps 4000 --batch-size 16 \
  --eval-every 500 --lr 0.0001 --max-seconds 900
python scripts/evaluate_crownless_facts.py --data MIXED --output STRICT_RESULTS \
  --device cpu --split strict_test --split strict_wording_test \
  --split editorial_test --model mixed=MIXED_RUN/best.ckpt
python scripts/evaluate_crownless_facts.py --data CORE_V2 --output GAME_RESULTS \
  --device cpu --split test --model mixed=MIXED_RUN/best.ckpt
make export_literary
./export_literary MIXED_RUN/best.ckpt MIXED_RUN/crownless-core-5m.litq8
```

The wider corpus's training and ordinary validation files were fixed before the strict name audit. Strict subsets were created from the training text and audit names before final evaluation. The builder now creates those strict subsets as part of the same command. Stored hashes identify the exact data and checkpoints used.

## Try speech

```sh
python scripts/speak_crownless.py --model MIXED_RUN/best.ckpt \
  --event '? ~ Displaced workers reinforce The Ditch Parliament on the old road.'
```

Repeat `--event` for older events first. Put the topic last. Use `?` for an uncertain account and `~` for one that has been widely retold. The sampler prints one completed spoken line.

Speech generation caches the event context. On the selected first-stage checkpoint, cached and full-context logits agreed within 0.00001. All 240 baseline test sentences matched between the original and cached evaluators. The model scores use floating-point weights with greedy decoding; the native quantized export receives a separate loading and speech smoke check.

Eight focused tests cover paired facts, uncertainty cues, name isolation, game replay, full-name filtering, corpus offsets, and cached decoding. Existing trainer and native C/PyTorch parity tests also pass. Raw samples, model hashes, and the two-stage training records accompany this report.
