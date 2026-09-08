# Subword and context experiments

Three runs use the same Gutenberg corpus and author-separated splits:

| Run | Parameters | Vocabulary | Context | Token presentations |
| --- | ---: | ---: | ---: | ---: |
| 5m-256 | 5,049,600 | 2,048 BPE | 256 | 100,007,936 |
| 5m-1024 | 5,049,600 | 2,048 BPE | 1,024 | 100,007,936 |
| 50m-1024 | 50,476,160 | 2,048 BPE | 1,024 | 800,006,144 |

The tokenizer learns from the training split only. Byte-level BPE preserves
all source text. Encoding processes lines with their original newlines. A full
round trip and byte-accounting check passed for every split. Training has
27,157,194 tokens, or 3.16 characters per token on average.

The small models have identical initial weights. Each update selects the same
8,192-token contiguous training block using a separate seeded generator. The
256-token model divides it into 32 sequences; the 1,024-token model divides it
into eight. Both see identical targets in the same order. The 50M run uses the
same selection rule with a larger token budget. Size and training budget both
change in that comparison.

The architecture retains ZERO's rotary attention, RMS normalization, GELU,
residual dropout and tied embeddings. Initialization uses normal standard
deviation 0.02, with residual output projections scaled by sqrt(2 * layers).
Training uses PyTorch AdamW, learning rate 0.0003, 2% warmup capped at 2,000
updates, cosine decay, dropout 0.1, weight decay 0.01 for matrices, and gradient
clipping at 1.0. CUDA uses BF16 autocast with FP32 weights and optimizer state.
These choices make the character baseline comparison a combined recipe change.

Selection scores 64 fixed windows every 100 updates. Final evaluation scores
1,024 fixed windows in each held-out split. Every window has the same 64 target
tokens across all models. Only the preceding context changes. Bits per byte
sums natural-log token losses and divides by log(2) and exact target bytes.
The character baseline is evaluated on those exact decoded target bytes.
This differs from the earlier 512-character-window evaluation. Samples use the
six existing prompts, seed 7, temperature 0.7, top-k 40 and 128 generated tokens.

One AWS g5.xlarge runs the three jobs in order, with a 12-hour limit including
setup. At the previously verified $1.006/hour compute price, the compute cap
is $12.072 plus storage and IPv4. The budget target is $14. Each run checks its
projected completion after 100 updates and saves its checkpoint first. Progress
uploads every two minutes. Final results upload before shutdown; shutdown
terminates the instance and deletes its encrypted disk.

Tests cover identical initial weights, causal masking, attention and gradient
agreement against an explicit attention reference, and identical evaluation
targets and byte counts. CPU smoke runs completed training, checkpoint save,
reload, evaluation and sampling for both small configurations. GPU throughput
and final quality are live results.

Tokenizer API reference: https://huggingface.co/docs/tokenizers/
