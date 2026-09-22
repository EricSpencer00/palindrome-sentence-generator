import json
from pathlib import Path

import experiments.paragraph_abcd_endpoint_transducer_20260921 as m


def test_endpoint_condition_and_complete_prose():
    result = m.run()
    assert result["stats"]["candidates"] == 16
    assert result["stats"]["exact"] == 0
    assert all(row["provenance"]["complete_sentence_templates"] for row in result["rendered_outputs"])
    assert all(row["live_transducer"]["outer_obligation"]["closed_before_interior"] for row in result["rendered_outputs"])


def test_independent_audit_and_shortcut_gates():
    result = m.run()
    for row in result["rendered_outputs"]:
        audit = row["audit"]
        assert audit["two_pointer_exact"] == audit["sha_equal"]
        assert not any(row["provenance"][key] for key in ("finished_text_reversal", "catalogue_text", "repeated_unit", "self_palindromic_unit", "fragment", "post_hoc_repair"))


def test_artifact_roundtrip():
    result = m.run()
    Path(m.OUT).write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(Path(m.OUT).read_text())["stats"] == result["stats"]
