# Crownless speech from the ZERO Gutenberg base

This pilot fine-tunes all 4,852,992 ZERO parameters on Crownless event lists and their spoken lines. The base is the Gutenberg checkpoint selected at update 98,500. The run used a local Apple M4 Max through PyTorch 2.8.0 MPS.

It learned the speech format and sentence endings. Generated speech still replaces supplied facts with familiar training facts. Every one of the six validation samples and sixteen separately written challenge samples contains a wrong named entity or a wrong main event on manual review. This checkpoint is an experiment; the next training-data revision should focus on preserving the supplied facts.

## Results

| Measure | Result |
| --- | ---: |
| Training examples | 4,000 |
| Validation / test examples | 500 / 500 |
| Separately written challenge examples | 16 |
| Updates / batch size | 1,000 / 8 |
| Validation loss before tuning | 1.9269 |
| Validation loss after tuning | 0.2378 |
| Test loss after tuning | 0.2546 |
| Challenge loss after tuning | 1.5443 |
| Speech-ending newline, validation samples | 6 / 6 |
| Speech-ending newline, challenge samples | 16 / 16 |
| Run time including evaluation | 106.6 seconds |

Loss is per output character with the reference prefix supplied at each step. This makes low loss compatible with wrong facts during free generation. All six base-model validation samples emitted an immediate newline under the same greedy decoding rule. After tuning, they produce complete sentences.

For example, the supplied event says Varkesh burns **Silverwick**. The tuned model says:

> Varkesh the Unappeased burned Gloamgate. The telling blamed stolen hoard money, if there's truth in the rumour.

A separately written food-shortage event about Bellmere produces a relief-charter notice about Thornford. The raw before, after, and challenge samples are included alongside the metrics.

## Reproduce

Use Python 3.11 with `torch==2.8.0` and `numpy==2.2.6`. Supply the base checkpoint and the Crownless v2 corpus. Their SHA-256 hashes are recorded in `metadata.json`. The base checkpoint hash is `9ea700a76eed50eb2f13e57cd00567e80a7e43449a562829ba74d35363021244`.

Crownless source: `atimics/crownlesscarriage` commit `79706fad39e95d8f20e763e277a42efb7b24896f`, PR https://github.com/atimics/crownlesscarriage/pull/636. Corpus settings: 96 worlds, 730 days each, first seed 1, 5,000 examples, at most 1,000 per event kind.

```sh
python scripts/test_tune_crownless.py
python -m unittest discover -s tests -p test_zero_torch.py
python scripts/tune_crownless.py --base BASE/best.ckpt --data CORPUS \
  --output RUN --device mps --steps 1000 --batch-size 8 --eval-every 100 \
  --lr 0.0001 --seed 17 --max-seconds 900
make export_literary
./export_literary RUN/best.ckpt RUN/crownless-5m.litq8
```

The trainer starts fresh Adam moments and computes loss only for the spoken line and its ending newline. It selects weights using validation loss. Final test and challenge loss are measured after selection. Model input is the plain event list. Audit files supply training boundaries and split checks.

The dataset has distinct worlds and exact spoken outputs across train, validation, and test. Their wording comes from shared generation rules. The challenge set adds separately authored wording and names. The six validation generations cover its six event kinds; they are a diagnostic sample. General English retention and a comparison against training from random weights remain open measurements.

All examples fit within the base model's 512-character context. The quantized export also loaded and generated text in the native runtime; `native-smoke.txt` records that smoke check. Native sampling differs from the greedy PyTorch evaluation above.

## Next data change

Generate many paired examples with the same event wording and different people, places, factions, and objects. Add pairs where only the final event changes. Vary the event wording as well as the spoken line. Hold out complete name sets and sentence forms. Score whether speech preserves the supplied people, places, event, and uncertainty.

Keep the grounded game gossip generator as the runtime baseline while measuring this trained model. The saved checkpoint provides a repeatable starting comparison for that next dataset.
