import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("lane", Path(__file__).with_name("bilateral_dialogue_cfg_20260922.py"))
lane = importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)

def test_bilateral_cursor_crosses_word_boundary():
    assert lane.bilateral_prefix("memos", "some m", width=5)["matched"] == 5
    assert not lane.bilateral_prefix("memos", "some men", width=5)["compatible"]

def test_typed_expansion_is_not_completed_sentence_bank():
    assert len(tuple(lane.expand(("det", "noun", "verb_past", "obj", ".")))) > 1
    assert lane.ROLES["A"][0][-1] == "."

def test_run_has_prose_provenance_and_independent_audits():
    result = lane.run()
    assert result["stats"]["slot_expansions"]["A"] > 1
    assert result["stats"]["candidate_completions"] > 0
    assert result["stats"]["exact_gt38"] == 0
    assert all(row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
               for row in result["rendered_candidates"])
    assert all(row["provenance"]["slot_expansion"] for row in result["rendered_candidates"])
    assert all("?" in row["roles"]["Q"] and "." in row["roles"]["R"] for row in result["rendered_candidates"])
    assert all("a inlet" not in row["rendered"].lower() for row in result["rendered_candidates"])
    assert all(row["roles"]["Q"].split()[0] in {"Did", "Could"}
               for row in result["rendered_candidates"])
