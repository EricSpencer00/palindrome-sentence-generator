"""Boundary and provenance checks for the saved-material realtime wrapper."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.realtime_chunk_construct import (ChunkConstructor, ROOT, exact_audit,
                                            letter_tape, theme_text)


class SavedChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constructor = ChunkConstructor(root=ROOT)

    def test_three_refrain_units_make_exact_abcba_and_keep_shared_words(self):
        result = self.constructor.construct(mode="refrain", target_letters=500,
                                             units=3, anchor="saw", seed=0)
        chunks = result["chunks"]
        self.assertEqual(result["status"], "constructed")
        self.assertEqual(len(chunks), 5)
        self.assertEqual([c["role"] for c in chunks],
                         ["opener", "opener", "turn", "return", "return"])
        self.assertEqual([c["text"] for c in chunks],
                         [c["text"] for c in reversed(chunks)])
        self.assertEqual(result["distinct_chunks"], 3)
        self.assertTrue(all("saw" in theme_text(c["text"]).split() for c in chunks))
        self.assertTrue(chunks[0]["text"].endswith("?"))
        self.assertTrue(chunks[2]["text"].endswith("."))
        tape = "".join(c.lower() for c in result["text"] if "a" <= c.lower() <= "z")
        self.assertEqual(tape, tape[::-1])
        self.assertFalse(result["novel"])

    def test_paired_rights_return_in_reverse_slot_order(self):
        result = self.constructor.construct(mode="paired", target_letters=500,
                                             units=3, seed=0)
        chunks = result["chunks"]
        self.assertEqual([c.get("pair") for c in chunks], [0, 1, 2, None, 2, 1, 0])
        for left, right in zip(chunks[:3], reversed(chunks[4:])):
            self.assertEqual(letter_tape(left["text"]), letter_tape(right["text"])[::-1])
            self.assertEqual(left["lineage"], right["lineage"])
        self.assertTrue(result["exactness"]["exact"])
        self.assertEqual(result["max_chunk_uses"], 1)
        self.assertEqual(result["human_readability"], "not evaluated")

    def test_short_targets_fail_without_clipping_any_chunk(self):
        for mode in ("refrain", "paired"):
            result = self.constructor.construct(mode=mode, target_letters=1, units=3)
            self.assertEqual(result["status"], "no_construction")
            self.assertEqual(result["chunks"], [])
            self.assertEqual(result["text"], "")
            self.assertFalse(result["exactness"]["exact"])

    def test_ceiling_reproducibility_and_no_adjacent_duplicate_chunks(self):
        for mode in ("refrain", "paired"):
            for target in (40, 100, 500):
                args = dict(mode=mode, target_letters=target, units=4, seed=7)
                result = self.constructor.construct(**args)
                self.assertEqual(result, self.constructor.construct(**args))
                self.assertLessEqual(result["letters"], target)
                self.assertFalse(result["adjacent_duplicate_chunks"])
                counts = Counter(letter_tape(c["text"]) for c in result["chunks"])
                self.assertLessEqual(max(counts.values(), default=0),
                                     2 if mode == "refrain" else 1)

    def test_ascii_audit_rejects_unsupported_letters_and_empty_tapes(self):
        for text in ("éve", "---", "not a palindrome"):
            self.assertFalse(exact_audit(text)["exact"])
        with self.assertRaises(ValueError):
            letter_tape("eve é")
        self.assertTrue(exact_audit("Never odd or even!\n")["exact"])

    def test_explicit_unknown_anchor_is_not_silently_replaced(self):
        result = self.constructor.construct(anchor="zzzzzz", target_letters=500)
        self.assertEqual(result["status"], "no_construction")

    def test_actual_reddit_title_keeps_its_text_and_credit(self):
        result = self.constructor.construct(anchor="elated", target_letters=500)
        self.assertEqual(result["status"], "constructed")
        self.assertEqual(result["letters"], 39)
        self.assertEqual(len(result["chunks"]), 1)
        sources = result["chunks"][0]["lineage"]
        post = next(s for s in sources
                    if s["metadata"].get("post_id") == "t3_1ut6vtp")
        self.assertEqual(post["metadata"]["author"], "u/Kind-Photograph6926")
        self.assertEqual(result["text"], post["original_text"])
        self.assertEqual(post["metadata"]["text"], post["original_text"])
        self.assertIn("/1ut6vtp/", post["metadata"]["source_url"])
        self.assertFalse(result["novel"])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_reddit = {
            "text": "Never odd or even!", "author": "u/example",
            "author_url": "https://www.reddit.com/user/example",
            "source_url": "https://www.reddit.com/r/palindromes/comments/example/",
            "post_id": "t3_example", "published": "2026-10-09T00:00:00Z",
            "observations": [{"file": "original-feed.json", "custom": [1, 2]}],
        }
        self.write("data/centres.json", ["was it a cat i saw"])
        self.write("data/canon_spelled.json", {"wasitacatisaw": "was it a cat i saw"})
        self.write("data/v3_bank.json", [])
        self.write("data/reddit-palindromes/2026-10-09/posts.json", {
            "credit": "Posters are credited; original authorship is not verified.",
            "rows": [self.original_reddit,
                     {"text": "Was it a cat I saw?", "author": "u/also_credited"},
                     {"text": "éve", "author": "u/unicode"},
                     {"text": "---", "author": "u/empty"},
                     {"text": "not a palindrome", "author": "u/nonexact"}],
        })
        self.write("tools/polaris/payload/brown.json.gz",
                   {"table": {}, "shapes": [], "trigrams": {}}, compressed=True)
        self.constructor = ChunkConstructor(root=self.root)

    def write(self, path, value, compressed=False):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        destination.write_bytes(gzip.compress(raw) if compressed else raw)

    def test_reddit_original_wording_credit_and_input_hash_survive(self):
        result = self.constructor.construct(anchor="never", target_letters=100, units=3)
        self.assertEqual(result["text"], self.original_reddit["text"])
        lineage = result["chunks"][0]["lineage"][0]
        self.assertEqual(lineage["metadata"], self.original_reddit)
        path = "data/reddit-palindromes/2026-10-09/posts.json"
        expected = hashlib.sha256((self.root / path).read_bytes()).hexdigest()
        self.assertEqual(lineage["input_sha256"], expected)
        self.assertEqual(result["source_hashes"][path], expected)
        self.assertFalse(result["novel"])
        # Responses cannot mutate the compiled source records for later calls.
        lineage["metadata"]["observations"][0]["custom"].append(3)
        next_result = self.constructor.construct(anchor="never", target_letters=100)
        self.assertEqual(next_result["chunks"][0]["lineage"][0]["metadata"], self.original_reddit)

    def test_tape_deduplication_keeps_every_source_credit(self):
        result = self.constructor.construct(anchor="saw", target_letters=100)
        self.assertEqual(len(result["chunks"]), 1)
        lineage = result["chunks"][0]["lineage"]
        self.assertEqual(len(lineage), 3)
        self.assertIn("u/also_credited", [r["metadata"].get("author") for r in lineage])
        self.assertEqual(self.constructor.inventory["counts"]["excluded_input_rows"], 3)
        self.assertEqual(self.constructor.inventory["dependencies"]["packages"], [])

    def test_overlength_centre_is_refused(self):
        result = self.constructor.construct(anchor="never", target_letters=2)
        self.assertEqual(result["status"], "no_construction")
        self.assertEqual(result["letters"], 0)


if __name__ == "__main__":
    unittest.main()
