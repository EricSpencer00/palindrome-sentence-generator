from experiments.character_trie_relative_decoder_20260920 import FRAMES, run, search


def test_relative_character_trie_exposes_internal_slots_without_false_exactness():
    rows, nodes = search(FRAMES[0], 39, max_nodes=2_000)
    assert nodes > 0
    assert rows == []


def test_relative_character_trie_run_is_independently_gated():
    result = run()
    assert result["stats"] == {
        "target_runs": 84,
        "nodes": 306725,
        "exact": 0,
        "mechanically_admitted": 0,
        "longest_exact_letters": 0,
    }
    assert result["independent_audit"] == [
        "two-pointer normalized tape",
        "forward/reverse SHA-256",
    ]
    assert result["novelty_preflight"]["catalogue_imported"] is False
    assert result["reader_gate"] == "closed; no exact candidate reached it"
