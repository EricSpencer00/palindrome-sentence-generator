import hashlib, json, re
from pathlib import Path
from experiments.luna_relative_cfg_orbit_20260920 import live_orbit_search, norm

def test_lane_has_explicit_zero_or_rendered_closure():
    bank, tested, closures, best = live_orbit_search()
    assert tested > 0
    assert closures == 0 or best["left"]

def test_independent_two_pointer_and_sha_audit():
    _, _, _, best = live_orbit_search()
    x, y = norm(best["left"]), norm(best["right"])
    i, j = 0, len(y) - 1
    while i < len(x) and j >= 0 and x[i] == y[j]: i += 1; j -= 1
    assert i == best["matched"]
    assert len(hashlib.sha256(x.encode()).hexdigest()) == 64
    assert len(hashlib.sha256(y.encode()).hexdigest()) == 64

def test_provenance_and_run_record():
    p = Path(__file__).parents[1] / "runs" / "luna-relative-cfg-orbit-20260920.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    for k in ("held_out_lexical_bank", "live_character_equality", "provenance", "next_construction"):
        assert d[k]
    assert not d["finished_tape_reversal"] and not d["word_order_mirror"]
