import hashlib
import json
from pathlib import Path

from experiments.typed_multiword_return_frames_20260922 import (
    FRAME_TYPES,
    Frame,
    equation_trace,
    independent_audit,
    render_pair,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "typed_multiword_return_frames_20260922.py"
ARTIFACT = ROOT / "runs" / "typed-multiword-return-frames-20260922.json"


def _set_a_date_pair():
    frame = Frame(
        frame_type="V_DET_N",
        words=("set", "a", "date"),
        tags=("vb", "at", "nn"),
        source="fixture",
        sentence=0,
        start=0,
        agreement="imperative-subject=you",
        completeness="complete transitive imperative",
    )
    return {
        "x": frame,
        "y": frame,
        "x_lemmas": ("set", "date"),
        "y_lemmas": ("set", "date"),
        "equation": equation_trace(frame, frame),
        "y_lane": "V_DET_N",
    }


def test_the_only_equation_hit_closes_online_but_fails_freshness():
    row = render_pair(_set_a_date_pair())
    assert row["equation"]["closed_before_render"] is True
    assert all(step["matched"] for step in row["equation"]["owner_residual_cursor_trace"])
    assert row["independent_audit"]["letters"] == 50 > 44
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["grammar"]["all_sentences_complete"] is True
    assert row["semantic_event_register"]["all_events_connected"] is True
    assert row["lemma_freshness"]["all_lemmas_distinct"] is False
    assert row["central_admission"]["distinct_words"] is False
    assert row["central_admission"]["no_repeated_nontrivial_unit"] is False
    assert row["mechanically_and_semantically_admitted"] is False


def test_return_frames_are_popped_in_strict_lifo_order():
    row = render_pair(_set_a_date_pair(), include_command_cycle=True)
    trace = row["return_stack"]["trace"]
    pushed = [step["words"] for step in trace if step["operation"] == "push_return_frame"]
    popped = [step["words"] for step in trace if step["operation"] == "pop_return_frame"]
    assert popped == list(reversed(pushed))
    assert row["independent_audit"]["letters"] == 58
    assert row["complementary_token_boundary_mask"]["passes"] is True
    assert row["proper_span_mask"]["passes"] is True


def test_remote_artifact_exhausts_only_the_four_declared_frame_lanes():
    payload = json.loads(ARTIFACT.read_text())
    lanes = payload["search"]["lanes"]
    assert [lane["frame_type"] for lane in lanes] == list(FRAME_TYPES)
    assert payload["fixed_conditions"]["bare_lemma_bank_widened"] is False
    assert payload["fixed_conditions"]["finished_palindromic_spans_excluded"] is True
    assert payload["stats"]["equation_pairs"] == 1
    assert payload["stats"]["rendered_survivors"] == 0
    assert payload["equation_pairs"][0]["x_words"] == ["set", "a", "date"]
    assert payload["equation_pairs"][0]["y_words"] == ["set", "a", "date"]
    for lane in lanes:
        assert lane["stats"]["typed_y_frames"] > 0
        if lane["frame_type"] != "V_DET_N":
            assert lane["stats"].get("complete_pairs", 0) == 0
        obstruction = lane["best_obstruction"]
        assert obstruction["owner"] in {"x_grammar_phase", "x_lexical_cursor"}
        assert isinstance(obstruction["cursor"], int)
        assert "unresolved_residual" in obstruction


def test_every_rendered_obstruction_is_independently_exact_and_explicit():
    payload = json.loads(ARTIFACT.read_text())
    assert len(payload["admission_obstructions"]) == 2
    for row in payload["admission_obstructions"]:
        audit = independent_audit(row["rendered"])
        assert audit == row["independent_audit"]
        assert audit["two_pointer_exact"] is True
        assert audit["hashes_agree"] is True
        assert row["failed_gates"] == ["global_lemma_freshness", "central_admission"]
        assert row["failed_central_checks"] == ["distinct_words", "no_repeated_nontrivial_unit"]


def test_artifact_records_reproducible_hst_bench_provenance():
    payload = json.loads(ARTIFACT.read_text())
    assert payload["provenance"]["host"] == "hst-bench"
    assert payload["provenance"]["python"] == "3.12.3"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == payload["provenance"]["source_sha256"]
    assert payload["corpus"]["brown_raw_sha256"] == (
        "0f7e73534a6aa2cabd8a685b5da7108f32a6e31df9b83509e7d8cd99b75a7de4"
    )
