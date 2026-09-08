# Inference and decoding study

The cache preserves the existing sliding-context semantics by rebuilding
when the prefix shifts. Original uncached sampling remains the default for
research callers; the live demo explicitly enables the cache.

A fixed decoding grid scores 12 settings: temperatures 0.6/0.7/0.8, top-k
20/40, and repetition penalties 1.0/1.1. Each setting uses the six fixed
prompts and generation seeds 1–8, for 576 outputs from the selected 50M
checkpoint. Every continuation has 128 new tokens. Repetition and diversity
are descriptive measures; blind coherence ratings require separate review.

The completed 16-run replication summary and successful finish record are
included here. The model card reports the paired training-seed intervals.

The demo offers 64/128/256 new tokens and optional display trimming at the
last detected sentence ending. This is a punctuation heuristic. It retains
full raw output, which the user can reveal with a checkbox. Samples with no
detected ending are displayed in full.
