from experiments.packed_staggered_paragraph_automaton_20260922 import (
    _sentences,
    grammars,
    run,
)


def test_packed_grammars_have_two_sentence_phases_on_both_sides():
    left, right = grammars()
    left_phases = {edge.phase for edges in left.values() for edge in edges}
    right_phases = {edge.phase for edges in right.values() for edge in edges}
    assert left_phases == {"A", "B"}
    assert right_phases == {"A-prime", "B-prime"}
    assert any(edge.word == "no" and edge.target == "Q"
               for edge in left["S"])
    assert any(edge.word == "no" and edge.target == "Q"
               for edge in right["S"])
    assert any(edge.word == "trace" and edge.phase == "B"
               for edge in left["Q"])


def test_witness_phase_groups_render_as_complete_sentences():
    assert _sentences(("an", "aide", "she", "waited"),
                      ("A", "A", "B", "B"), ("A", "B")) == (
        "An aide.", "She waited."
    )


def test_small_packed_run_preserves_reader_gate():
    result = run(max_states=1_000, max_results=5)
    assert result["novelty_preflight"]["word_bank_widening"] is False
    assert result["novelty_preflight"]["intermediate_word_closure_rejected"] is True
    assert "states_reaching_both_second_sentences" in result["stats"]
    assert result["reader_packet"] == []
    assert all(row["mechanically_admitted"]
               for row in result["mechanically_admitted_candidates"])
