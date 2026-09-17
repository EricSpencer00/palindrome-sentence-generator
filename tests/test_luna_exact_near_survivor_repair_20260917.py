import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).parents[1]
def tape(s): return ''.join(re.findall('[A-Za-z]',s)).lower()
def test_held_out_paired_slot_repairs_are_independently_exact():
    run=json.loads((ROOT/'runs/luna-exact-near-survivor-repair-20260917.json').read_text())
    assert run['novelty_preflight']['passed']
    assert run['stats']=={'base_letters':101,'mutations':3,'exact':3,'mechanically_admitted':0}
    for row in run['rows']:
        t=tape(row['rendered']); a=row['exact_audit']
        assert a['two_pointer_exact'] and a['sha_equal']
        assert a['sha256_forward']==hashlib.sha256(t.encode()).hexdigest()
        assert a['sha256_reverse']==hashlib.sha256(t[::-1].encode()).hexdigest()
        assert row['provenance']['catalogue_imported'] is False
        assert row['provenance']['reversed_finished_sentence'] is False
        assert row['readability_status'].startswith('not certified')
        assert row['next_repair']
