from experiments.bounded_character_cfg_intersection_20260921 import audit, intersect


def test_constructive_intersection_audits_cleanly():
    result = audit(intersect())
    assert result["rendered"] == "Able was I ere I saw Elba."
    assert result["frontier_empty"]
    assert result["two_pointer"]
    assert result["sha_equal"] is True  # independent hashes agree because the string is a palindrome.
    assert result["provenance"]
    assert result["shortcut_gate"]


def test_no_repeated_unit_shortcut():
    result = intersect()
    words = result["rendered"].rstrip(".").lower().split()
    assert words != words[: len(words) // 2] * 2
