import json

from experiments.incumbent_666_typed_discourse_lattice_20260922 import (
    DIRECT_LEFT,
    DIRECT_RIGHT,
    DIRECT_CHILD_SHA256,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_typed_lattice_persists_zero_closure_and_changed_seam_exact_child():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "typed-lattice-direct-seam-666")
    lattice = payload["lattice_attempt"]
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert lattice["stats"]["closures"] == 0
    assert lattice["stats"]["relation_candidates"] > 0
    assert lattice["stats"]["character_contradictions"] > 0
    assert lattice["deepest_prefix_obstruction"]["cursor"] is not None
    assert lattice["deepest_prefix_obstruction"]["residual"]
    assert row["promotion_status"]["promoted"] is False
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == DIRECT_CHILD_SHA256
    assert result["sha_equal"]
    assert normalize(DIRECT_LEFT) == normalize(DIRECT_RIGHT)[::-1]


def test_lattice_uses_relation_and_template_gates():
    payload = json.loads(OUT.read_text())
    lattice = payload["lattice_attempt"]

    assert "because/so/after/when/then" in lattice["relation_gate"]
    assert "identical subject-verb" in lattice["template_gate"]
    assert lattice["stats"]["template_pair_rejections"] > 0
    assert payload["changed_seam_after_zero_closure"]["normalized_left"] == [127, 197]
    assert payload["changed_seam_after_zero_closure"]["normalized_right"] == [469, 539]
