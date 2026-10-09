"""Small verifier/prompt tests; no Torch, model loading, or GPU work."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.argonne_model_meaning_proposals_20261009 import (
    PUBLIC_POSTS, PUBLIC_SNAPSHOT_ROOT, ROOT, SOURCE_SPECS,
    atomic_checkpoint, audit_candidate, build_prompt, choose_cache, load_public_inputs, normalized_letters,
    parse_response, proposal_cases, summarize,
)


class ProposalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "short", "arm": "direct", "target_letters": 11,
                     "min_letters": 11, "max_letters": 11, "intent": "A person owns a home.",
                     "reference": None}

    def test_ascii_exactness_length_and_unicode_rejection(self):
        result = audit_candidate("Emo had a home!", self.case, {}, {})
        self.assertTrue(result["whole_text_exact"])
        self.assertTrue(result["eligible_exact_proposal"])
        self.assertEqual(result["letters"], 11)
        self.assertFalse(audit_candidate("Emo had a home! ß", self.case, {}, {})["whole_text_exact"])
        self.assertFalse(audit_candidate("!!! 123", self.case, {}, {})["whole_text_exact"])
        self.assertEqual(normalized_letters("É A ß"), ("a", ["É", "ß"]))

    def test_reference_duplicate_and_wrong_length_are_separate(self):
        tape, _ = normalized_letters("Emo had a home")
        reference = audit_candidate("EMO HAD A HOME.", self.case, {tape: ["credited"]}, {})
        duplicate = audit_candidate("Emo had a home", self.case, {}, {tape: 0})
        self.assertTrue(reference["whole_text_exact"])
        self.assertFalse(reference["eligible_exact_proposal"])
        self.assertEqual(reference["reference_matches"], ["credited"])
        self.assertEqual(duplicate["duplicate_of_attempt"], 0)
        self.assertFalse(duplicate["eligible_exact_proposal"])
        wrong = audit_candidate("Emo had a home", {**self.case, "min_letters": 12}, {}, {})
        self.assertFalse(wrong["length_pass"])

    def test_feedback_contains_actual_positions_and_same_intent(self):
        text = "Emo had a house."
        audit = audit_candidate(text, self.case, {}, {})
        first = audit["first_mismatches"][0]
        self.assertEqual((first["left_offset"], first["right_offset"]), (1, 10))
        self.assertEqual((first["left_char"], first["right_char"]), ("m", "s"))

    def test_feedback_prompt_and_invalid_response_retention(self):
        text = "Emo had a house."
        audit = audit_candidate(text, self.case, {}, {})
        previous = {"attempt": 0, "candidate": text, "audit": audit}
        prompt = build_prompt(self.case, 0, previous)
        self.assertIn(self.case["intent"], prompt)
        self.assertIn('"left_offset"', prompt)
        self.assertIn('"letters": 12', prompt)
        self.assertIn(text, prompt)
        self.assertEqual(parse_response("not JSON"), (None, "invalid_json"))
        self.assertEqual(parse_response('{"text":"x","plot":"p","meaning":3}')[1], "expected_exact_text_plot_object")
        parsed, error = parse_response(json.dumps({"text": text, "plot": "A proposed plot"}))
        self.assertIsNone(error)
        self.assertEqual(parsed["text"], text)

    def test_public_inputs_all_arms_and_credit(self):
        sources, demo, references, hashes = load_public_inputs()
        self.assertEqual([source["letters"] for source in sources], [568, 666, 752])
        self.assertGreaterEqual(demo["letters"], 80)
        self.assertEqual(demo["author"], "u/catinore")
        self.assertEqual(len(hashes), 4)
        cases = proposal_cases(sources, demo)
        self.assertEqual({case["arm"] for case in cases}, {"direct", "reconstruction", "phrase_overhang"})
        self.assertEqual([case["target_letters"] for case in cases[:3]], [120, 240, 480])
        self.assertEqual([(case["min_letters"], case["max_letters"]) for case in cases[:3]],
                         [(120, 239), (240, 479), (480, 959)])
        self.assertEqual([(case["min_letters"], case["max_letters"]) for case in cases[3:6]],
                         [(568, 1135), (666, 1331), (752, 1503)])
        direct_prompt = build_prompt(cases[0], 0)
        self.assertIn("at least 120", direct_prompt)
        self.assertNotIn("Use exactly", direct_prompt)
        phrase = cases[-1]
        self.assertEqual((phrase["min_letters"], phrase["max_letters"]), (40, 160))
        self.assertIn(demo["source_url"], build_prompt(phrase, 0))
        self.assertFalse(phrase["independent_novelty_claim"])
        self.assertIn("reddit:" + demo["post_id"], references[normalized_letters(demo["text"])[0]])
        self.assertEqual(summarize([])["meaning_measurements"], 0)

    def test_existing_task_cache_is_reused_and_missing_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, reused = choose_cache(root, root)
            self.assertEqual(cache, root.resolve())
            self.assertTrue(reused)
            with self.assertRaises(FileNotFoundError):
                choose_cache(root, root / "missing")

    def test_interrupted_attempt_checkpoint_preserves_input_without_response(self):
        row = {"attempt": 0, "status": "generating", "prompt": "Frozen intended plot",
               "seed": 7, "case_id": "direct-120", "raw_response": None,
               "generated_tokens": None, "audit": None, "parse_error": None,
               "proposal_stage": "initial_proposal"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            atomic_checkpoint(path, {"rows": [row], "stats": summarize([row])})
            prepared = json.loads(path.read_text())
            self.assertEqual(prepared["stats"]["attempts_started"], 1)
            self.assertEqual(prepared["stats"]["responses_returned"], 0)
            row["status"] = "interrupted"
            atomic_checkpoint(path, {"rows": [row], "stats": summarize([row])})
            interrupted = json.loads(path.read_text())
            self.assertEqual(interrupted["rows"][0]["prompt"], "Frozen intended plot")
            self.assertIsNone(interrupted["rows"][0]["raw_response"])
            self.assertIsNone(interrupted["rows"][0]["generated_tokens"])
            self.assertEqual(interrupted["stats"]["interrupted"], 1)
            self.assertEqual(interrupted["stats"]["responses"], 0)
        legacy = {key: value for key, value in row.items() if key != "status"}
        self.assertEqual(summarize([legacy])["responses_returned"], 1)

    def test_scoped_snapshots_load_when_all_originals_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for spec in SOURCE_SPECS:
                original = ROOT / spec["path"]
                source = original if original.exists() else ROOT / PUBLIC_SNAPSHOT_ROOT / spec["path"]
                destination = root / PUBLIC_SNAPSHOT_ROOT / spec["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            posts = root / PUBLIC_POSTS
            posts.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / PUBLIC_POSTS, posts)
            sources, demo, _, hashes = load_public_inputs(root)
            self.assertEqual([source["sha256"] for source in sources],
                             [spec["sha256"] for spec in SOURCE_SPECS])
            self.assertTrue(all(source["resolved_source_path"].startswith(PUBLIC_SNAPSHOT_ROOT + "/")
                                for source in sources))
            self.assertEqual(demo["author"], "u/catinore")
            self.assertEqual(len(hashes), 4)
            # A present, changed original cannot be bypassed by the snapshot.
            changed = root / SOURCE_SPECS[0]["path"]
            changed.parent.mkdir(parents=True, exist_ok=True)
            changed.write_text(json.dumps({"rows": [{"id": SOURCE_SPECS[0]["row_id"],
                                                      "rendered": "changed"}]}))
            with self.assertRaisesRegex(ValueError, "public_source_changed_or_invalid"):
                load_public_inputs(root)


if __name__ == "__main__":
    unittest.main()
