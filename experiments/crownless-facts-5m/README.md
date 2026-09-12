# Crownless paired-fact pilot

The first Crownless model learned speech patterns while replacing supplied names with familiar ones. This run uses the same 4,852,992-parameter ZERO Gutenberg base and a larger set of varied facts. It trains on event lists followed by one spoken line.

## Data

The builder extracts 415 patterns from the Crownless v2 training audit. These preserve the game's existing spoken wording, uncertainty threshold of 40, and retelling threshold of four. Added input forms express the same event facts. The original game corpus is the source for the shared wording.

There are 24,000 training examples across twelve topics: notices, deaths, dragon fire, dragon warnings, goblin raids, shortages, harvests, route closures, peace, treasures, bandits, and cult recruitment. Each topic receives 2,000 examples. There are 3,346 distinct training names across people, places, factions, dragons, and objects.

Each pair changes one named fact in the final event. Earlier event lines stay the same. Training includes one, two, and three event lines. Both members of every pair stay in one split. The examples are synthetic variations of simulation events; names and facts are reassigned for training.

Validation has 240 examples with reserved names. The final name test has another 240 examples. A separate 240-example test uses a third input form reserved for evaluation in each topic. Full names are distinct across training, validation, and test. Names share ordinary English parts. The wording test draws names from the test pool.

The original sixteen editorial examples are kept unchanged as a regression check. They were inspected in the first pilot and remain a small known challenge set. The automated measures count exact required-name matches, changed-name responses across pairs, exact reference sentences, and sentence endings. Required-name matches alone measure one part of factual accuracy. Event meaning and uncertainty also need review.

## Reproduce

Use Python 3.11, `torch==2.8.0`, and `numpy==2.2.6`. The base model is the Gutenberg checkpoint selected at update 98,500. Its SHA-256 is `9ea700a76eed50eb2f13e57cd00567e80a7e43449a562829ba74d35363021244`.

Supply the Crownless v2 corpus produced from `atimics/crownlesscarriage` commit `79706fad39e95d8f20e763e277a42efb7b24896f`. The source hashes are recorded in `corpus-manifest.json`.

```sh
python scripts/build_crownless_facts.py --source CORE_V2 --output FACTS
python scripts/test_crownless_facts.py
python scripts/tune_crownless.py --base GUTENBERG/best.ckpt --data FACTS \
  --output RUN --device mps --steps 6000 --batch-size 16 --eval-every 500 \
  --lr 0.0002 --max-seconds 900
python scripts/evaluate_crownless_facts.py --data FACTS --output COMPARISON \
  --device mps --model first=FIRST_PILOT/best.ckpt --model revised=RUN/best.ckpt
make export_literary
./export_literary RUN/best.ckpt RUN/crownless-facts-5m.litq8
```

The run starts fresh optimizer moments and tunes all core weights. Validation loss selects the checkpoint. The reserved test sets are used after selection. Evaluation uses greedy decoding, printable ASCII, a 160-character speech limit, and the first newline as the sentence boundary. A blank line counts as an empty result.

## Try a spoken line

```sh
python scripts/speak_crownless.py --model RUN/best.ckpt \
  --event '? Flintwing burns Willowford because 17 stolen crowns remain missing.'
```

Repeat `--event` for older events first. Put the topic last. Prefix an uncertain event with `?` and a widely retold event with `~`. The sampler prints one spoken line.

## Validation

Six focused tests cover name separation, the game's uncertainty boundary, exact-name boundaries, wrong-name detection in pairs, deterministic corpus generation, pair integrity, and agreement between batched and single-example decoding. The existing trainer tests and native C/PyTorch checkpoint parity test also pass.
