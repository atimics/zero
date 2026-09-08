# Gutenberg corpus for ZERO

This corpus gives ZERO varied English fiction for language training. Braid builds
the release. The input list covers adult fiction, adventure stories, children's
stories, and fantasy from 29 authors. The build requires at least ten million
training words and 100,000 words in each held-out split.

## Files

- `sources.json`: the selected Gutenberg catalog entries and mirror URLs.
- `sources.lock.json`: exact source hashes, cleaned hashes, word counts, and splits.
- `build-summary.json`: measured final sizes and Braid release identity.
- `../../scripts/build_gutenberg_corpus.py`: download, prepare, and compile steps.

The raw books and large output files live in the chosen output directory. The
release includes source text, metadata, duplicate decisions, and Braid checksums.
The `ready` directory contains `train.txt`, `validation.txt`, and `test.txt`, plus
JSONL versions with a source record for each chunk.

## Source selection and rights

The first release selects up to eight books per author from a saved Gutenberg
catalog. It selects English fiction with one named contributor, and filters
collected editions and repeated normalized titles. `sources.json` is the fixed
input list; a later catalog update leaves this list intact.

The selected original English authors died by 1946. Source records retain each
edition's title, author, catalog link, and download link. Raw files retain the
Gutenberg license and credits. The cleaner checks the body markers and flags
copyright notices in the edition header for review. The source classification
is based on the selected public-domain editions; Braid checks that declared
classification. Edition and territory rights should be checked for each future
addition, including translators and editors.

Catalog: <https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz>

Mirror list: <https://www.gutenberg.org/MIRRORS.ALL>

Download policy: <https://www.gutenberg.org/policy/robot_access.html>

Permission guide: <https://www.gutenberg.org/policy/permission.html>

## Build

Use Python 3.10+, Node 22+, Git, and pnpm 10.19.0. Prepare Braid at this commit:

```sh
git clone https://github.com/cenetex/braid.git /tmp/braid-zero
git -C /tmp/braid-zero checkout 0b8d9bb66d83e8c2dfd75daff4db5a48d911237f
cd /tmp/braid-zero
corepack pnpm install --frozen-lockfile --ignore-scripts
corepack pnpm build
```

From the ZERO repository, run:

```sh
python3 scripts/build_gutenberg_corpus.py download --output /tmp/zero-gutenberg
python3 scripts/build_gutenberg_corpus.py prepare --output /tmp/zero-gutenberg
python3 scripts/build_gutenberg_corpus.py compile --output /tmp/zero-gutenberg --braid /tmp/braid-zero
```

Downloads use Gutenberg's listed PGLAF mirror and pause two seconds between
books. Existing files are reused. The committed source lock catches upstream
edition changes. Preserve raw downloads to reproduce this exact release after
Gutenberg changes an edition. A new corpus version should use a new lock path.

Braid uses stable inline source IDs so release identity is independent of the
output directory. It chunks at paragraph boundaries around 24,000 characters,
applies quality gates, and removes exact and near duplicate chunks across the
whole collection. Near-duplicate checking compares all eligible chunks in
memory; larger collections will need a more efficient index.

## Splits and evaluation

The SHA-256 hash of a fixed seed and normalized author/title assigns each work
to training (18 of 20 buckets), validation (one bucket), or test (one bucket).
The assignment happens before chunking. Every surviving chunk from a work stays
in its assigned split. Word proportions vary with book lengths.

The final checks cover disjoint work groups, unique exact chunk hashes, the
Braid release checksums, the fixed split rule, and ASCII output. Near-duplicate
matching is an approximate filter; passage-level review remains useful for
anthologies and alternate editions.

Use `ready/train.txt` as language-training input. ZERO's current `--text` reader
reserves the final 5% of each input internally. The separate validation and test
files here support a future explicit held-out scoring step. Keep those two files
separate from training inputs.

This release teaches literary language. Dialogue records and Warrenmind examples
form a later training stage. Model trials should measure sentence quality,
relevant answers, story consistency, response endings, and browser speed.

## Checks

```sh
python3 -m unittest discover -s tests -p test_gutenberg_corpus.py
```

The small tests cover wrapper removal, ASCII conversion, incomplete downloads,
rights notices, fixed splits, and changed source bytes. The full build also runs
Braid's release verification before writing its summary.
