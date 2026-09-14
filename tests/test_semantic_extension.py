from llm_palindrome.semantic_extension import SemanticFrame, extend_fixed_center, instantiate
from llm_palindrome.validator import is_palindrome, normalize


def test_frame_instantiation_is_stable_and_grammatical_slots_are_preapplied():
    frame = SemanticFrame("event", (("we", "i"), ("read", "write"), ("notes", "maps")))
    assert list(instantiate(frame)) == ["we read notes", "we read maps", "we write notes", "we write maps",
                                        "i read notes", "i read maps", "i write notes", "i write maps"]


def test_exact_gate_happens_after_semantic_product():
    frame = SemanticFrame("toy", (("ab", "ba"),))
    rows = extend_fixed_center("aba", [frame], min_letters=1)
    assert rows == ["ab aba ba", "ba aba ab"]
    assert all(is_palindrome(row) for row in rows)
    assert len(normalize(rows[0])) == 7


def test_length_gate_rejects_short_exact_closures():
    frame = SemanticFrame("toy", (("ab", "ba"),))
    assert extend_fixed_center("aba", [frame], min_letters=8) == []
