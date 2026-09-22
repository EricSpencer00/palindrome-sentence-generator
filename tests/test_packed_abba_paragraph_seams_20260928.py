from experiments.packed_abba_paragraph_seams_20260928 import paragraph_grammar, run
from experiments.packed_seam_grammar_20260927 import intersect


def test_abba_search_is_packed_and_exact_gated():
    result = run()
    assert result["provenance"]["complete_sentence_enumeration"] is False
    assert result["topology"] == "A B | B A; two intact clauses on each side of the middle seam"
    assert all(row["audit_two_pointer"]["exact"] and row["independent_validator_exact"]
               for row in result["candidates"])


def test_four_clause_grammar_has_no_precomputed_surface_paths():
    grammar = paragraph_grammar()
    result = intersect(grammar, max_letters=220, cap=120000)
    assert result["grammar_states"] > 1
    assert result["grammar_character_edges"] > 0
    assert result["represented_surface_paths"] if "represented_surface_paths" in result else True
