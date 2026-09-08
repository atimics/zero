#!/usr/bin/env python3
"""Build ZERO's book corpus with the pinned Braid compiler (Python standard library)."""
import argparse
import hashlib
import json
import re
import subprocess
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAID_COMMIT = "0b8d9bb66d83e8c2dfd75daff4db5a48d911237f"
SEED = "zero-gutenberg-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def split_for(work_group):
    bucket = int(digest((SEED + "\0" + work_group).encode())[:8], 16) % 20
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def clean_book(raw):
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*[^\n]*\n", text, re.M)
    end = re.search(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.M)
    if not start or not end or start.end() >= end.start():
        raise ValueError("Gutenberg body markers missing or reversed")
    if re.search(r"(?:copyright|permission of the copyright holder)", text[:start.start()], re.I):
        raise ValueError("Edition requires a separate rights review")
    text = text[start.end():end.start()]
    text = re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.I)
    for source, target in {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "..."}.items():
        text = text.replace(source, target)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # ZERO consumes 128-character ASCII. Preserve paragraph and dialogue layout.
    text = "".join(c if c == "\n" or 32 <= ord(c) <= 126 else " " for c in text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)
    result = "\n\n".join(paragraphs) + "\n"
    if len(result.split()) < 1000:
        raise ValueError("Book has fewer than 1000 cleaned words")
    return result


def download(args):
    books = json.loads(args.sources.read_text())["books"]
    args.output.joinpath("raw").mkdir(parents=True, exist_ok=True)
    locked = {}
    if args.lock.exists():
        locked = {b["id"]: b for b in json.loads(args.lock.read_text())["books"]}
    failures = []
    for index, book in enumerate(books):
        path = args.output / "raw" / f"{book['id']}.txt"
        try:
            if not path.exists():
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(book["url"], headers={"User-Agent": "ZERO-corpus/1.0"})
                        with urllib.request.urlopen(request, timeout=45) as response:
                            data = response.read(20_000_001)
                        if len(data) > 20_000_000:
                            raise ValueError("Book exceeds download limit")
                        path.write_bytes(data)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** (attempt + 1))
                time.sleep(args.delay)
            data = path.read_bytes()
            if book["id"] in locked and digest(data) != locked[book["id"]]["raw_sha256"]:
                raise ValueError("Downloaded edition differs from sources.lock.json")
            clean_book(data)
            print(f"{index + 1}/{len(books)} acquired {book['id']}: {book['title'].splitlines()[0]}", flush=True)
        except Exception as error:
            failures.append({"id": book["id"], "error": str(error)})
            print(f"{index + 1}/{len(books)} failed {book['id']}: {error}", flush=True)
    write_json(args.output / "download-failures.json", failures)
    if failures:
        raise SystemExit("Resolve download failures before preparing the corpus")


def prepare(args):
    books = json.loads(args.sources.read_text())["books"]
    old_lock = json.loads(args.lock.read_text()) if args.lock.exists() else None
    records, entries = [], []
    for book in books:
        raw = (args.output / "raw" / f"{book['id']}.txt").read_bytes()
        text = clean_book(raw)
        split = split_for(book["work_group"])
        entry = {**book, "split": split, "raw_sha256": digest(raw),
                 "clean_sha256": digest(text.encode()), "words": len(text.split()),
                 "license_basis": "Project Gutenberg public-domain edition; named original English author"}
        entries.append(entry)
        records.append({"text": text, "book_id": book["id"], "title": book["title"],
                        "author": book["author"], "work_group": book["work_group"],
                        "split": split, "source_url": book["url"],
                        "raw_sha256": entry["raw_sha256"]})
    lock = {"version": 1, "seed": SEED, "braid_commit": BRAID_COMMIT, "books": entries}
    if old_lock is not None and lock != old_lock:
        raise ValueError("Source lock differs; use a new lock path for a new corpus version")
    write_json(args.lock, lock)
    args.output.joinpath("books.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    spec = {"apiVersion": "braid/v1", "kind": "DatasetBuild",
            "metadata": {"name": "zero-gutenberg", "version": "v1",
                         "description": "English fiction for ZERO. Book splits precede chunking."},
            "spec": {"purposes": ["pretraining", "research"],
                     # Stable inline document IDs keep release identity independent of local paths.
                     "sources": [{"id": f"pg-{book['id']}", "kind": "inline",
                                  "documents": [{"id": str(book["id"]), "text": record["text"],
                                                 "metadata": {k: v for k, v in record.items() if k != "text"}}],
                                  "license": "public-domain", "attribution": book["author"] + "; " + book["catalog_url"],
                                  "domain": book["domain"], "language": "en"}
                                 for book, record in zip(books, records)],
                     "rights": {"allowedLicenses": ["public-domain"], "requireAttribution": True},
                     "chunking": {"targetCharacters": 24000, "overlapCharacters": 0},
                     "quality": {"minimumCharacters": 512, "nearDuplicateHammingDistance": 3},
                     "selection": {"strategy": "ranked", "seed": SEED},
                     "evaluation": {"minimumDocuments": 1000, "maximumRejectionRatio": 0.15},
                     "output": {"directory": "releases"}, "publication": {"target": "none"}}}
    write_json(args.output / "braid.json", spec)
    print(f"Prepared {len(books)} books, {sum(b['words'] for b in entries):,} words", flush=True)


def compile_corpus(args):
    actual = subprocess.check_output(["git", "-C", str(args.braid), "rev-parse", "HEAD"], text=True).strip()
    if actual != BRAID_COMMIT:
        raise ValueError(f"Braid must be at {BRAID_COMMIT}")
    dirty = subprocess.check_output(["git", "-C", str(args.braid), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Braid has modified tracked files")
    cli = args.braid / "dist" / "cli.js"
    subprocess.run(["node", str(cli), "build", str(args.output / "braid.json")], check=True)
    manifests = list((args.output / "releases").glob("zero-gutenberg/v1/*/release.json"))
    if len(manifests) != 1:
        raise ValueError("Use an output directory with exactly one Braid release")
    release = manifests[0].parent
    subprocess.run(["node", str(cli), "verify", str(release)], check=True)
    export_corpus(args.output, release, args.minimum_train_words)


def export_corpus(output, release, minimum_train_words):
    records = [json.loads(line) for line in (release / "data/train.jsonl").read_text().splitlines()]
    split_records = {split: [] for split in ["train", "validation", "test"]}
    membership, seen_hashes = {}, set()
    for record in records:
        metadata = record["metadata"]
        split = metadata["split"]
        group = metadata["work_group"]
        if group in membership and membership[group] != split:
            raise ValueError("Work occurs in multiple splits")
        membership[group] = split
        if record["contentHash"] in seen_hashes:
            raise ValueError("Duplicate chunk survived Braid")
        seen_hashes.add(record["contentHash"])
        if split != split_for(group):
            raise ValueError("Split differs from the fixed seed")
        split_records[split].append(record)
    summary = {"version": 1, "seed": SEED, "braid_commit": BRAID_COMMIT,
               "release_id": json.loads((release / "release.json").read_text())["releaseId"],
               "split_unit": "normalized author and work title", "splits": {}}
    ready = output / "ready"
    ready.mkdir(exist_ok=True)
    for split, items in split_records.items():
        items.sort(key=lambda r: (r["metadata"]["book_id"], r["metadata"].get("chunkIndex", 0)))
        text = "\n\n".join(r["text"] for r in items) + "\n"
        data = text.encode("ascii")
        words = len(text.split())
        if words < (minimum_train_words if split == "train" else 100_000):
            raise ValueError(f"{split} has only {words:,} words")
        (ready / f"{split}.txt").write_bytes(data)
        (ready / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
        summary["splits"][split] = {"words": words, "characters": len(data), "chunks": len(items),
                                   "books": len({r['metadata']['book_id'] for r in items}),
                                   "text_sha256": digest(data)}
    summary["checks"] = {"work_groups_disjoint": True, "exact_chunk_hashes_unique": True,
                         "braid_release_verified": True, "ascii_only": True}
    write_json(ready / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "prepare", "compile"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=ROOT / "corpus/gutenberg/sources.json")
    parser.add_argument("--lock", type=Path, default=ROOT / "corpus/gutenberg/sources.lock.json")
    parser.add_argument("--braid", type=Path)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--minimum-train-words", type=int, default=10_000_000)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "compile" and args.braid is None:
        parser.error("compile requires --braid")
    {"download": download, "prepare": prepare, "compile": compile_corpus}[args.stage](args)


if __name__ == "__main__":
    main()
