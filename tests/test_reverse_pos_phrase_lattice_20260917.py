from experiments.reverse_pos_phrase_lattice_20260917 import run


def test_reverse_conditioned_phrase_lattice_preserves_near_misses_and_zero_exact():
    artifact = run()
    assert artifact["stats"]["phrase_count"] == 21
    assert artifact["stats"]["rendered_rows"] == 26
    assert artifact["stats"]["exact_rows"] == 0
    assert artifact["provenance"]["seed_words_used"] is False
    assert all(not row["audit"]["exact"] for row in artifact["candidates"])


def test_best_phrase_pair_is_diagnostic_not_reader_candidate():
    artifact = run()
    best = artifact["candidates"][0]
    assert best["text"] == "the old man sat the pot was hot"
    assert best["audit"]["letters"] == 24
    assert best["audit"]["exact"] is False
