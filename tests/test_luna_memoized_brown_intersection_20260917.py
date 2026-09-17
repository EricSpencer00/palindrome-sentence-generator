import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).parents[1]
def t(s): return "".join(re.findall(r"[A-Za-z]",s)).lower()
def test_memoized_heldout_lane_has_prose_and_independent_audits():
    r=json.loads((ROOT/"runs/luna-memoized-brown-intersection-20260917.json").read_text())
    assert r["novelty_preflight"]["passed"] and r["novelty_preflight"]["held_out_shapes"]==3
    assert r["stats"]=={"shapes":3,"frames":4,"rendered":12,"exact":0,"mechanically_admitted":0,"max_letters":63,"memo_states":12}
    assert len({x["shape"] for x in r["rows"]})==3
    for x in r["rows"]:
        q=t(x["rendered"]); a=x["exact_audit"]
        assert len(q)==a["letters"] and q
        assert a["sha256_forward"]==hashlib.sha256(q.encode()).hexdigest()
        assert a["sha256_reverse"]==hashlib.sha256(q[::-1].encode()).hexdigest()
        assert x["provenance"]["borrowed_text"] is False
        assert x["semantic_roles"].startswith("agent=")
        assert x["next_repair"]
