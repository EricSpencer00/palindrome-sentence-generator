from experiments.character_trie_grammar_decoder_20260919 import FRAMES, run


def test_character_trie_recovers_anchor_without_posthoc_reversal():
    result = run()
    assert result["stats"]["exact"] == 1
    assert result["stats"]["longest_exact_letters"] == 38
    row = result["candidates"][0]
    assert row["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
    assert row["provenance"]["finished_tape_reversed"] is False
    assert row["provenance"]["catalogue_imported"] is False
    assert any("D" in frame and "S" in frame and "V" in frame for frame in FRAMES)
