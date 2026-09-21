import json
from pathlib import Path

import support_driven_paired_character_grammar_20260921 as m


def test_trace_matches_rendered_normalized_tape():
    result = m.run()
    for row in result["rendered_candidates"]:
        left = m.letters(" ".join(m.words(m.Frame(**row["left_frame"]))))
        right = m.letters(" ".join(m.words(m.Frame(**row["right_frame"]))))[::-1]
        assert row["support_depth"] <= min(len(left), len(right))
        assert row["audit"]["letters"] == len(left) + len(right)


def test_no_false_exact_or_connector_seam():
    result = m.run()
    assert result["stats"]["paired_states"] == 36
    assert result["stats"]["exact_candidates"] == 0
    assert all("while" not in row["rendered"] for row in result["rendered_candidates"])
    assert all(row["gates"]["independent_pointer_hash"] is False for row in result["rendered_candidates"])


def test_artifact_roundtrip():
    result = m.run()
    Path(m.OUT).write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(Path(m.OUT).read_text())["stats"] == result["stats"]
