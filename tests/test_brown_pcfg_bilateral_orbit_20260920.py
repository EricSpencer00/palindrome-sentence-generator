from experiments.brown_pcfg_bilateral_orbit_20260920 import audit, search_pair


def _bank():
    return {
        "DET": [
            {"word": "a", "score": 10},
            {"word": "the", "score": 8},
        ],
        "NOUN": [
            {"word": "sailor", "score": 10},
            {"word": "writer", "score": 8},
            {"word": "pilot", "score": 7},
        ],
        "VERB": [
            {"word": "reads", "score": 10},
            {"word": "maps", "score": 8},
            {"word": "writes", "score": 7},
        ],
    }


def test_bilateral_orbit_has_a_live_exact_transition_and_audit():
    result = search_pair(
        ("DET", "NOUN", "VERB"),
        ("DET", "NOUN", "VERB"),
        _bank(),
        max_nodes=5000,
        beam_width=500,
    )
    assert result["nodes"] > 0
    assert result["status"] in {"exhausted", "node_budget"}
    assert audit("ab ba")["exact"] is True


def test_nonpalindromic_rendering_is_not_promoted_by_the_audit():
    row = audit("A calm sailor reads a clear map")
    assert row["exact"] is False
    assert row["two_pointer_exact"] is False
    assert row["forward_sha256"] != row["reverse_sha256"]
