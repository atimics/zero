# Matched dialogue study

Two equal-budget local runs from the shipped 4,935,937-parameter model:

- Small: twelve human-approved conversations.
- Broad: 46 usable conversations from the 48-draft batch, including the approved twelve. Two source packets are outside the current parser.

Both avatars hold the same account throughout training and evaluation. This covers the game's shared-account conversation case. Each turn sees the prior four spoken lines with relative self/other labels. Training includes authored opening wording and the shipped model's opening wording. Partial first-name references are expanded to the full known name so they use existing field-copy slots.

Each arm starts from the same checkpoint and seed 913, uses 1,200 steps, batch size sixteen, AdamW learning rate 0.0001, and four original grounded opening rows per batch. The other twelve rows are dialogue. Checkpoint choice is the fixed last step.

Fresh evaluation worlds 1601 and 1602 each run for 365 days. Evaluation uses sixteen source accounts chosen before baseline generation. Both arms exclude all 46 training accounts from evaluation after case folding and numeral normalization. Names and event grammar may still overlap; this is fine-tune transfer within known simulation rules.

Assess the generated conversations for relevant replies, facts, uncertainty, and language. Track termination, exact opening preservation, repeated lines, and replacement characters as diagnostics. Target-fit scores use a fixed every-other-row sample and measure recall separately. No result threshold will trigger automatic game integration.
