from __future__ import annotations

from experiments.morphology_first_dependency_lattice_20260916 import (
    STATE_SPACE_SIGNATURE,
    audit,
    novelty_preflight,
    run,
)


def test_novelty_preflight_is_before_generation_and_collision_free():
    result = novelty_preflight()
    assert result["status"] == "novel_exact_signature"
    assert result["exact_signature_collision"] == []
    assert result["artifact_collision"] == []
    assert "lemma-derivation-inflection-path" in STATE_SPACE_SIGNATURE


def test_morphology_lattice_preserves_derivation_and_independent_audit():
    result = run()
    assert result["stats"]["raw_yields"] == 1728
    assert result["stats"]["exact_yields"] >= 2
    candidate = next(row for row in result["exact_candidates"] if "metallic sonatas" in row["rendered"])
    assert candidate["audit"]["exact"] is True
    assert candidate["morphology_path"][3]["derivation"] == ["-ic"]
    assert candidate["morphology_path"][4]["inflection"] == ["plural"]
    assert candidate["audit"]["normalized_tape"] == candidate["audit"]["normalized_tape"][::-1]
    assert audit(candidate["rendered"])["exact"] is True
