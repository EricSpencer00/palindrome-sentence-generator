from experiments.half_tape_grammar_csp_20260919 import FRAMES, _audit, _options, _place, run


def test_half_tape_alias_rejects_conflicting_character_assignments():
    # In a four-letter target positions 0 and 3 alias the same variable.
    assert _place([None, None], 0, ("ab",), 4) is not None
    assert _place(["a", None], 3, ("b",), 4) is None


def test_half_tape_pilot_records_exact_rows_with_independent_audits():
    result = run(lengths=range(38, 41), max_nodes=30_000)
    assert result["stats"]["exact"] >= 1
    assert result["stats"]["longest_exact"] >= 38
    assert result["exact_candidates"]
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
        assert row["audit"]["normalized"] == row["audit"]["normalized"][::-1]


def test_shared_participant_temporal_repair_is_agreement_gated():
    frame = next(frame for frame in FRAMES if frame.name == "shared_participant_temporal")
    assert "COREF" in frame.chunks
    assert {option.words for option in _options("COREF", {"subject_number": "sg"})} == {("she",), ("he",)}
    assert {option.words for option in _options("COREF", {"subject_number": "pl"})} == {("they",), ("we",)}


def test_relative_complement_is_a_distinct_agreement_path():
    frame = next(frame for frame in FRAMES if frame.name == "relative_complement_then_second_beat")
    assert frame.chunks[:4] == ("SUBJ", "VERB", "OBJ", "RELMARK")
    assert {option.words for option in _options("RELMARK", {})} == {("who",), ("that",)}
    assert all(option.number == "sg" for option in _options("RELVERB", {"relative_number": "sg"}))
    assert all(option.number == "pl" for option in _options("RELVERB", {"relative_number": "pl"}))
