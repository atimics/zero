# Crownless creature and emotion pilot

This experiment trains the existing 4,935,937-parameter conversation core on
three creature identities (human, goblin, pony) and three feelings (calm,
afraid, relieved). Each held account and spoken history appears in all nine
combinations. Confidence remains an independent account field.

## Training contract

The encoder adds `voice: goblin; feeling: afraid` before the held-account cue.
These are explicit controls supplied by the caller. The existing tokenizer,
source-copy mechanism, model shape, and four-message history stay in use.
Examples without performance controls follow the original encoding path.

This first pilot uses two authored openings per creature and two endings per
feeling around the original response. These teach and measure control of
wording. Richer character writing and acoustic performance have separate
listening and writing reviews. The style scores below count those authored
forms; they measure a narrower skill than a person's judgement of character.

The data contains 10,800 styled examples from 1,200 account rows and 1,200
ordinary conversation examples. Each training batch contains eight of each.
Training uses the published conversation core, AdamW, learning rate 0.0001,
50 warm-up steps, decay, clipping at one, and seed 91. Validation loss chooses
the saved checkpoint. Every score uses the exported 8-bit model.

Whole event kinds are held out of style training: alliances, dragon patrons,
horse breeding, peace, and settlement raids. The base model already learned
these meanings. This test measures their combination with the new styles.
The original corpus separates full names across training and test.

The first pilot runs 1,000 steps. A 3,000-step follow-up tests longer training
after the first run showed confusion between relief and calm. The earlier
test sets are development checks for that follow-up. A separate evaluation
uses account pairs excluded from all earlier development checks and changes
the question wording. Both runs and the fresh evaluation retain raw outputs.

## Reproduce

Install `scripts/requirements-crownless-v2.txt`. Generate the held-account
corpus with Crownless's `tools/build_core_diagnostic.py`, seed 20260919.
The corpus grammar must match the published base model.

```sh
python scripts/test_crownless_performance.py
python scripts/train_crownless_performance.py --data /path/to/core-data --output /tmp/performance-pilot --steps 3000 --device mps
python scripts/evaluate_crownless_performance_fresh.py --pilot /tmp/performance-pilot --data /path/to/core-data --output /tmp/performance-fresh
python scripts/review_crownless_performance.py --pilot /tmp/performance-pilot --output /tmp/performance-review
```

Use `--device cpu` or `--device cuda` for those systems. The output directory
must be fresh. Manifests record source, model, tokenizer, grammar, dataset,
and split hashes. The published evidence uses gzip to keep the review small.

## Use the trained model

Export a packet with Crownless's `core_account_probe --packet` interface.
The packet carries `grammar_sha256`, the held text, confidence, selected rule,
and field offsets. Optional history is a JSON list of `{speaker, text}` objects
with relative `self` and `other` speaker labels.

```sh
python scripts/speak_crownless_performance.py --model models/crownless-performance/core.ccv2 --packet /tmp/account.json --creature goblin --emotion afraid
```

The matching `performance.json` receipt identifies the model. The existing
Crownless C/WASM encoder needs the same explicit performance prefix before
this experimental checkpoint can be adopted by the game. This package supplies
the Python runner and review evidence for that later integration.

## Listen

The review page presents one account across all nine combinations, plus three
four-turn exchanges. Exchanges pass each model-generated line to the next
speaker. Their open conversation quality is for human review.

Audio uses one fixed synthetic reference per creature across all three moods.
Qwen VoiceDesign creates the references. Pocket TTS speaks the saved model's
exact words. Emotion reaches Pocket through the wording. This experiment tunes
the language core; the audio weights come from the existing speech models.

Use the existing separate Qwen and Pocket environments and cached models:

```sh
python scripts/audio_crownless_performance.py --mode design --references /tmp/performance-references --cache /path/to/qwen-cache --device mps
python scripts/audio_crownless_performance.py --mode speak --references /tmp/performance-references --review /tmp/performance-review --cache /path/to/pocket-cache
python -m http.server 8879 --bind 127.0.0.1 --directory /tmp/performance-review
```

The audio tool uses local cached weights. Receipts record text, seeds,
reference and audio hashes, package versions, model snapshots, and timing.
Listening should judge clear words, stable identity, and audible emotion.
