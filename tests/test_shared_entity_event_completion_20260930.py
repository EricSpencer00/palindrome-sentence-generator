from experiments.shared_entity_event_completion_20260930 import (
    EVENTS, compile_event_graph, run,
)
from experiments.packed_seam_grammar_20260927 import norm


def test_reference_event_requires_completed_antecedent():
    grammar, trace = compile_event_graph(("praise_mural", "make_mural"))
    assert grammar is None
    assert trace == []


def test_forward_event_completion_is_typed_and_online():
    grammar, trace = compile_event_graph(("make_mural", "praise_mural"))
    assert grammar is not None
    assert trace[-1]["required"] == ["artist", "mural"]
    assert trace[-1]["introduced_before"] == ["artist", "mural"]
    assert grammar.count > 1


def test_run_records_independent_gate_and_provenance():
    result = run()
    valid = [c for c in result["conditions"] if c.get("valid_typed_order")]
    assert len(valid) == 1
    assert result["provenance"]["complete_sentence_enumeration"] is False
    assert result["provenance"]["per_candidate_rlaif"] is False
    for row in valid[0]["candidates"]:
        assert row["audit"]["exact"]
        assert row["audit"]["independent_validator_exact"]
        assert row["forward_sha256"] == row["reverse_sha256"]
        assert row["novel_relative_to_seed"]
