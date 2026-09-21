from experiments.abba_connector_first_trie_20260922 import run, letters


def test_connector_first_lane_records_independent_audits_and_no_shortcut():
    result = run()
    assert result["stats"]["relations"] == 3
    assert result["stats"]["branches"] == 12
    assert result["stats"]["rendered_candidates"] == 0
    assert result["stats"]["exact_gt38"] == 0
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(c["audit"]["letters"] > 0 for c in result["controls"])
    assert all("sha256_forward" in c["audit"] and
               "sha256_reverse_obligation" in c["audit"] for c in result["controls"])
    assert all(c["provenance"]["relation_selected_before_surface"] for c in result["controls"])


def test_connector_obligation_is_letter_level():
    result = run()
    for cert in result["residual_certificates"]:
        assert cert["residual_prefix"]
        assert all(ch.isalpha() for ch in cert["residual_prefix"])
