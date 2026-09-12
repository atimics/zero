# Crownless performance core

An experimental 4,935,937-parameter conversation model with explicit human,
goblin, and pony identities and calm, afraid, and relieved feelings.
The model uses this repository's MIT license and the existing tokenizer in
`models/crownless-core-v2/tokenizer.json`.

The artifact has 5,044,766 bytes and SHA256
`fde6ce97344e6aaa9e681a58cd11319ad70d946be75532d6f115488098ed5eca`.
`performance.json` binds the model, tokenizer, grammar, and input contract.
Use `scripts/speak_crownless_performance.py` to supply controls with a native
held-account packet and optional recent speech.

Fresh evaluation: 548/648 complete approved responses with the requested
style, with all 648 creature and emotion forms matched. The
[experiment report](../../experiments/crownless-performance/README.md) explains
the authored style conventions, response errors, training, and listening review.

This model uses the performance prefix implemented in the Python encoder.
Game adoption requires adding that same control contract to the C/WASM encoder.
