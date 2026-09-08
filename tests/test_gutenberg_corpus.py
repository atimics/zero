import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("corpus", Path(__file__).resolve().parents[1] / "scripts/build_gutenberg_corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CorpusTests(unittest.TestCase):
    def raw(self, body=None):
        body = body or ('“The boat,” said Alice, “is ready.”\n\n' * 200)
        return ("Title: Example\r\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\r\n" +
                body + "\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nLicense wrapper").encode()

    def test_body_and_ascii(self):
        text = corpus.clean_book(self.raw())
        self.assertTrue(text.isascii())
        self.assertIn('"The boat,"', text)
        self.assertNotIn("License wrapper", text)
        self.assertNotIn("Title: Example", text)

    def test_incomplete_download(self):
        with self.assertRaises(ValueError):
            corpus.clean_book(self.raw().split(b"*** END")[0])

    def test_rights_notice(self):
        with self.assertRaises(ValueError):
            corpus.clean_book(b"Copyright 2025\n" + self.raw())

    def test_split_membership(self):
        groups = [f"author {i}|work" for i in range(1000)]
        forward = {g: corpus.split_for(g) for g in groups}
        reverse = {g: corpus.split_for(g) for g in reversed(groups)}
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward.values()), {"train", "validation", "test"})
        self.assertEqual(corpus.split_for("author|story"), corpus.split_for("author|collected stories"))

    def test_lock_detects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            raw = root / "raw/1.txt"
            raw.write_bytes(self.raw())
            sources = root / "sources.json"
            sources.write_text(json.dumps({"books": [{"id": 1, "title": "Example", "author": "Author",
                "work_group": "author|example", "url": "https://example.org/1.txt",
                "catalog_url": "https://example.org/1", "domain": "fiction"}]}))
            args = SimpleNamespace(output=root, sources=sources, lock=root / "lock.json")
            corpus.prepare(args)
            first = (root / "braid.json").read_bytes()
            corpus.prepare(args)
            self.assertEqual(first, (root / "braid.json").read_bytes())
            raw.write_bytes(self.raw().replace(b"Alice", b"Mara"))
            with self.assertRaises(ValueError):
                corpus.prepare(args)

    def test_export_rejects_author_overlap(self):
        first = corpus.split_for("author|story")
        second = "test" if first != "test" else "train"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            records = [{"metadata": {"work_group": f"author|work {i}", "split": split},
                        "contentHash": str(i)} for i, split in enumerate([first, second])]
            (root / "data/train.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            with self.assertRaisesRegex(ValueError, "Author occurs"):
                corpus.export_corpus(root, root, 0)

    def test_export_rejects_duplicate_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            record = {"metadata": {"work_group": "author|story", "split": corpus.split_for("author|story")},
                      "contentHash": "same"}
            (root / "data/train.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Duplicate chunk"):
                corpus.export_corpus(root, root, 0)


if __name__ == "__main__":
    unittest.main()
