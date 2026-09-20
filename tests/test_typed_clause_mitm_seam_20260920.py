from experiments.typed_clause_mitm_seam_20260920 import audit, main

def test_mitm_has_seam_controls_and_independent_audits():
    result=main()
    assert result['novelty_preflight']['status']=='passed'
    assert result['stats']['joins']>0
    assert result['complete_prose_controls']
    for row in result['complete_prose_controls']:
        assert row['audit']==audit(row['rendered'])
        assert row['provenance']['live_character_seam']
        assert not row['provenance']['repeated_units']
