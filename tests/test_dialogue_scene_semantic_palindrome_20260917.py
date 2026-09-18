from experiments.dialogue_scene_semantic_palindrome_20260917 import (
    BASE_EDGES,
    _surface,
    independent_audit,
    run,
)


def test_base_scene_is_independently_exact_and_rendered():
    text = _surface(BASE_EDGES)
    audit = independent_audit(text)
    assert text == (
        "Noel, now live on; Damon, draw a map. Was I sore? Eros: "
        "I saw Pam, a ward. Nomad: no evil won, Leon."
    )
    assert audit["letters"] == 68
    assert audit["exact"] is True
    assert audit["mismatch_count"] == 0
    assert audit["sha256_forward"] == audit["sha256_reverse"]


def test_boundary_shift_is_exact_but_not_word_order_mirror():
    text = _surface(BASE_EDGES, repair=True)
    audit = independent_audit(text)
    assert audit["exact"] is True
    assert "Pam award Nomad" in text
    # This repair changes the lexical segmentation rather than repeating the
    # original word-pair surface.
    from llm_palindrome.admission import mechanical_admission_checks
    checks = mechanical_admission_checks(text, min_letters=30, max_letters=200)
    assert checks["not_word_order_symmetry"] is True
    assert checks["exact_letter_palindrome"] is True


def test_run_records_longer_diagnostic_and_keeps_reader_gate_closed(tmp_path):
    result = run(tmp_path / "dialogue.json")
    assert result["summary"]["longest_exact_letters"] == 100
    assert result["summary"]["human_readability_claim"] is False
    assert result["summary"]["mechanically_reader_eligible"] == 0
    assert all(row["audit"]["exact"] for row in result["candidates"])
    assert result["candidates"][1]["parent_sha256"] == result["candidates"][0]["audit"]["sha256_forward"]
    assert "center-crossing" in result["next_repair"]
