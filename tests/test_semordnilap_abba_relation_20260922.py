from experiments.semordnilap_abba_relation_20260922 import run, letters


def test_lane_emits_complete_controls_and_independent_audits():
    data = run()
    assert data["stats"]["complete_AB_controls"] == 20
    assert data["stats"]["rendered_candidates"] == 40
    assert data["stats"]["exact_shortcut_clean_gt38"] == 0
    assert all("sha256_forward" in row["audit"] and "sha256_reverse_obligation" in row["audit"]
               for row in data["rendered_candidates"])


def test_semordnilap_lane_does_not_claim_reversal_shortcut():
    data = run()
    assert data["novelty_preflight"]["not_finished_tape_reversal"]
    assert all(row["provenance"]["finished_tape_reversal"] is False
               for row in data["rendered_candidates"])
    assert all(len(letters(row["rendered"])) > 38 for row in data["rendered_candidates"])
