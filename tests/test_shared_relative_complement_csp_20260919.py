from experiments.shared_relative_complement_csp_20260919 import run, search


def test_shared_relative_run_is_bounded_and_audited():
    result = run(range(40, 61), max_nodes=20_000)
    assert result["stats"]["nodes"] > 0
    assert result["provenance"]["rlaif_per_candidate"] is False
    assert result["reader_gate"] == "closed"


def test_shared_participant_path_has_live_grammar_state():
    result = search(60, max_nodes=20_000)
    for row in result["actual_candidates"]:
        assert len(row["word_path"]) == 7
        assert row["provenance"]["finished_tape_reversed"] is False
