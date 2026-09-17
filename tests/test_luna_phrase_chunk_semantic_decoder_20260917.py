import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "luna_phrase_chunk_semantic_decoder_20260917",
    ROOT / "experiments" / "luna_phrase_chunk_semantic_decoder_20260917.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
novelty_preflight, run = MODULE.novelty_preflight, MODULE.run


def test_phrase_chunk_frontier_emits_complete_nonshortcut_prose():
    result = run()
    assert result["novelty_preflight"]["passed"]
    assert result["candidate_count"] > 0
    assert result["exact_count"] == 0
    for row in result["rendered_candidates"]:
        assert row["letters"] >= 100
        assert row["independent_reparse"]
        assert row["independent_exact_agreement"]
        assert not row["anti_shortcut_flags"]["word_order_mirror"]
        assert not row["anti_shortcut_flags"]["repeated_chunk"]
        assert not row["anti_shortcut_flags"]["fragment"]
        assert row["next_repair"]


def test_phrase_chunk_novelty_preflight_has_no_collision():
    assert novelty_preflight()["collisions"] == []
