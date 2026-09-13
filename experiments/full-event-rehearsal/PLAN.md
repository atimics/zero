# Full-event rehearsal

Both arms train the same 296 rows derived from 46 conversations and two opening-wording choices. Both speakers retain their held account, with four previous spoken lines. The narrow control rehearses the 46 original openings. The full arm rehearses 3,660 opening rows spanning all 61 original meanings and earlier conversation responses for meanings outside the new dialogue coverage.

Each arm starts from the shipped model with seed 914, 1,200 steps, batch size sixteen, twelve dialogue rows and four rehearsal rows, and AdamW learning rate 0.0001. The final step determines the checkpoint. The full arm expands both the meaning coverage and the response types in rehearsal; this comparison measures their combined effect.

Sixteen new world accounts from seeds 1801 and 1802 measure four-turn conversation. They are separate from the new dialogue examples after case folding and numeral normalization. A fixed first case per meaning from the diagnostic builder's test split measures opening retention across all 61 meanings. The original grammar's accepted wording forms provide a narrow factual check; other wording needs review.

Review the actual conversations, opening facts, repeated replies, and uncertainty. Keep the deployed game model unchanged during the study. Training examples include authored rule substitutions and assistant-written dialogue; twelve dialogue scenes have human approval.
