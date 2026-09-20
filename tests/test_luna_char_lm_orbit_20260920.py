from experiments.luna_char_lm_orbit_20260920 import letters, search, independent_audit

def test_live_beam_has_rendered_candidates_and_audit():
    expanded, rows = search(limit=25)
    assert expanded > 0 and rows
    independent_audit(rows)
    assert all("text" in r and r["letters"] == len(letters(r["text"])) for r in rows)

def test_audit_does_not_promote_near_miss():
    _, rows = search(limit=5)
    independent_audit(rows)
    for r in rows:
        assert r["audit"]["exact"] == (letters(r["text"]) == letters(r["text"])[::-1])
