import hashlib
from experiments.semantic_frame_product_20260926 import audit, live_match, run, tape

def test_independent_pointer_and_hash_audit():
    s='An aide rips nine memos; some men inspire Diana.'; a=audit(s); t=tape(s)
    assert a['two_pointer_exact'] and all(t[i]==t[-1-i] for i in range(len(t)))
    assert a['sha256_forward']==hashlib.sha256(t.encode()).hexdigest()
    assert a['sha256_reverse']==hashlib.sha256(t[::-1].encode()).hexdigest()

def test_frame_product_is_online_and_records_provenance():
    p=run(40, 20)
    assert p['experiment']=='semantic-frame-product-20260926'
    assert p['calibration']['audit']['two_pointer_exact']
    assert p['calibration']['generated'] is False
    assert p['next_construction']
    for row in p['candidates']:
        assert row['provenance']['live_character_invariant']
        assert row['provenance']['no_finished_tape_reversal']

def test_live_match_checks_exposed_characters_without_reversal():
    ok, compared = live_match(['an', 'aide'], ['Diana', 'some'])
    assert not ok and compared > 0
    ok, compared = live_match(['ab'], ['ba'])
    assert ok and compared == 2
