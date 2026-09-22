import json
from pathlib import Path

from experiments.fixed_tape_surface_lattice_20260922 import build_artifact


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "fixed-tape-surface-lattice-20260922.json"


def test_scope_is_exactly_the_two_fixed_tapes() -> None:
    result = build_artifact()
    targets = {row["source_id"]: row for row in result["targets"]}

    assert set(targets) == {"target-42", "target-44"}
    assert len(targets["target-42"]["normalized_tape"]) == 42
    assert len(targets["target-44"]["normalized_tape"]) == 44
    assert targets["target-42"]["expected_sha256"] == (
        "1013004f658bdefeaaf7dea69c6d90a5d2c53381dbb5d7290ab7b25a8e5de1c3"
    )
    assert targets["target-44"]["expected_sha256"] == (
        "96462b7bc06958668e9d13d13ebd0b63e4f682a51939c5bc34d7aa230d39b104"
    )


def test_every_variant_preserves_tokens_tape_and_sha() -> None:
    result = build_artifact()
    for target in result["targets"]:
        ranks = []
        for variant in target["variants"]:
            ranks.append(variant["diagnostic_rank"])
            assert variant["same_frozen_word_tokens"]
            assert variant["same_normalized_tape"]
            assert variant["audit"]["two_pointer_exact"]
            assert variant["audit"]["sha256_forward"] == target["expected_sha256"]
            assert variant["ordinary_english_parse_attempt"]
            assert variant["exact_parse_obstruction"]
            assert not variant["reader_certified"]
            assert not variant["eligible_for_blinded_packet"]
        assert ranks == list(range(1, len(ranks) + 1))


def test_negative_result_does_not_emit_a_reader_packet() -> None:
    result = build_artifact()
    assert not result["decision"]["preserve_new_rendering"]
    assert not result["decision"]["emit_blinded_variant_packet"]
    assert all(
        target["materially_clearer_nonfragmentary_variant"] is None
        for target in result["targets"]
    )
    assert "never a readability" in result["ranking_policy"]


def test_checked_in_artifact_matches_source() -> None:
    assert json.loads(ARTIFACT.read_text()) == build_artifact()
