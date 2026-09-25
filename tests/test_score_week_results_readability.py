from experiments.score_week_results_readability import (
    select_brown_window,
    verify_candidate,
)


def test_exact_audit_checks_the_rendered_letters_and_manifest_hash():
    candidate = {
        "id": "control",
        "surface": "An aide rips nine memos; some men inspire Diana.",
        "letters": 38,
        "normalized_sha256": "ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6",
        "independently_exact": True,
        "metrics": {"word_count": 9},
    }
    audit = verify_candidate(candidate)
    assert audit["letters"] == 38
    assert all(audit["checks"].values())


def test_exact_audit_rejects_a_changed_surface():
    candidate = {
        "id": "changed-control",
        "surface": "An aide rips nine memos; some men inspire Dianx.",
        "letters": 38,
        "normalized_sha256": "ce71723a3eab38613adeb89c3ce18bab20286d91e6bcee20b25d3f4a724184c6",
        "independently_exact": True,
        "metrics": {"word_count": 9},
    }
    try:
        verify_candidate(candidate)
    except ValueError as exc:
        assert "changed-control failed exactness audit" in str(exc)
    else:
        raise AssertionError("changed manifest surface was accepted")


def test_brown_control_windows_match_word_count_and_avoid_used_spans():
    sentences = [
        (0, ["one", "two"]),
        (1, ["three"]),
        (2, ["four", "five"]),
        (3, ["six"]),
    ]
    first, end, count = select_brown_window(sentences, 3, [], "first")
    assert count == 3
    first2, end2, count2 = select_brown_window(
        sentences, 3, [(first, end)], "second")
    assert count2 >= 1
    assert end2 <= first or end <= first2
