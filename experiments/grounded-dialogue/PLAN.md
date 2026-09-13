# Grounded dialogue expansion

Build explicit field-copy targets into a dialogue bank covering the shipped model's 61 meanings. Each exchange has an opening, a short question, an answer, and a reaction. The bank is assistant-authored and awaits human review. It uses the earlier review's preference for brief, concrete exchanges.

Compare two continuations from the same shipped 4.94M base: the earlier 46-scene set with full rehearsal, and the new 61-meaning bank with opening rehearsal. Both use seed 915, 2,400 steps, batch sixteen (twelve dialogue and four rehearsal), AdamW learning rate 0.0001, CPU, and the fixed final checkpoint. This measures copying and coverage as a combined data change.

Both speakers hold the account and read previous generated speech. Before training, freeze sixteen accounts from fresh simulation worlds 1901 and 1902. Read all resulting exchanges. Also evaluate the first generated test case for each meaning, with fresh names and fixed questions; change individual source fields to check whether answers follow the input. Score copy actions and text separately from writing quality. Include both confident and uncertain contexts. Test questions come from the bank, so this is an in-grammar grounding test.

Use Python int8 reloads for evaluation. Keep the game deployment separate. Publish source, tests, complete test outputs, data hashes, candidate hashes, and reproduction steps in a PR.
