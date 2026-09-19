from experiments import large_lexicon_cfg_orbit_20260920 as lane

def test_scalable_cfg_trie_and_live_audit():
    out = lane.run(40)
    assert out["novelty_preflight"]["status"] == "novel"
    assert out["stats"]["lexicon_words"] >= 70
    assert out["stats"]["states"] == 40
    assert out["provenance"]["word_boundaries_before_render"]
    assert out["provenance"]["live_orbit_assignment"]
    for row in out["controls"]:
        assert row["trie_intersection"]["left_words"]
        assert row["orbit_assignment"]
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]

def test_roles_are_complete_and_no_repair_flags():
    assert all(lane.LEXICON[k] for k in ("det", "subject", "finite", "object", "prep", "place"))
    out = lane.run(3)
    p = out["provenance"]
    assert not p["post_hoc_repair"] and not p["finished_tape_reversal"] and not p["word_order_mirror"]
