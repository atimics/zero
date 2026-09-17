# Crownless corpus lineage

The corpora themselves are hundreds of megabytes of generated JSONL and are not
in git. What is here is the chain that produces them: every manifest pins the
hash of the one above it, so a corpus can be rebuilt and checked rather than
trusted. This exists because the last crownless corpus was lost — it lived only
in an untracked `out/` directory and its generating commit had been squashed
away.

## The chain

    crownless sim (crownless@2bd00aa3)
      └─ tools/build_core_diagnostic.py --pairs 12500
           → core-accounts-12500        manifest 98c16f51a313ff8c…
      └─ tools/build_core_mind.py
           → core-mind-balanced         manifest ed32824aea0a64da…
             └─ scripts/build_crownless_acts.py  --seed 73 --repeats 2
                  → crownless-acts-v1   (thirteen acts)
                    └─ scripts/train_crownless_acts.py
                         → acts-v1, sha256 216750d9971bac86…  ← shipped
              └─ scripts/build_crownless_moves.py --seed 73 --repeats 2
                   → crownless-moves-v2  (move × channel × stance)
                     └─ scripts/train_crownless_moves.py
                          → no accepted checkpoint yet
              └─ scripts/build_crownless_moves.py --seed 73 --repeats 2
                 --holdout-cells <12 move:voice:stress triples, one per move>
                 --confusable-replacements
                   → crownless-moves-v3  (holdouts + near-miss swaps)
                     └─ scripts/train_crownless_moves.py --typed-stance
                          → Run A: unseen-cell accuracy gates the upgrade
              └─ scripts/build_crownless_moves.py --seed 73 --repeats 2
                 --holdout-cells <same 12> --confusable-replacements --situation
                   → crownless-moves-v4  (situation bits, no wording signal)
                     └─ scripts/train_crownless_moves.py --typed-stance --situation
                          → B1 attempt HELD: hungry 27%, sheltered 27%,
                             in_transit 17% — bits decorrelated from targets
              └─ v4 + SITUATION_MARKS (predicament openings, 18 lines)
                   → crownless-moves-v5
                     └─ scripts/train_crownless_moves.py --typed-stance --situation
                          → B1 retry: guard 47/48, all three axes ≥30%
              └─ v5 + SOCIAL_MARKS (debts/trust/faction/distance, 36 lines)
                 + --social → crownless-moves-v6
                   └─ scripts/train_crownless_moves.py --typed-stance
                       --situation --social
                        → B2: guard 45/48; owes/trusts/faction/far all emit;
                          goal/courage found target-silent (courage conditions
                          no pool anywhere — pre-existing, not a regression)
              └─ v6 + STANCE_MARKS (courage and goal openings, 30 lines)
                 + copy-span offset now locates the account in the target
                   → crownless-moves-v7
                     └─ scripts/train_crownless_moves.py --typed-stance
                         --situation --social
                          → B3: guard 45/48, copies 12/12; goal 55%,
                            courage 75% any-change (were 17% each), stress 70%

Both corpus manifests name the accounts and mind manifests they were built
from, by hash; both resolve against the files above. `acts-v1.results.json`
records what the shipped checkpoint scored on its own test, wording, bridge and
chat sets.

## Staleness (2026-09-16)

`tools/build_core_mind.py` in crownless no longer writes `# goal:`, `# stress:`
or `# courage:` prefix lines: stance travels as row fields and the typed encoder
carries it on the meta channel. `core-mind-balanced` and everything below it
were built by the older builder, so a rebuild from current sources will not
match the manifests pinned here. That is the check working as designed, not a
silent drift — but do not train a typed run on the old rows expecting the model
to read stance text it was never given. Rebuild the chain before reuse.

## Rebuilding

Deterministic given the same inputs and `--seed 73 --repeats 2`. The account and
mind corpora come from the crownless repository; the act and move corpora from
`scripts/build_crownless_*.py` here. Check a rebuild by comparing its
`manifest.json` against the copy in this directory.

## Durable storage

Still unsolved. BRAID is the project's corpus layer, but it is a rights-aware
document pipeline: `file` and `directory` sources are flattened to a text field
per document, which would discard the prefix, field spans, copy spans and
accepted-wording pools these rows exist to carry. A synthetic corpus derived
from the simulator also has no external rights to track, which is most of what
BRAID's specification asks for. Pinned private object storage fits the artifact
better than a BRAID collection does.
