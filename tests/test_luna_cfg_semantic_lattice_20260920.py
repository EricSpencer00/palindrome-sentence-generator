import hashlib
import importlib.util
import sys
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/luna_cfg_semantic_lattice_20260920.py"
spec = importlib.util.spec_from_file_location("lane", P)
lane = importlib.util.module_from_spec(spec); sys.modules[spec.name] = lane; spec.loader.exec_module(lane)

def test_lattice_renders_independent_scene_paths():
    rows = lane.main()["candidates"]
    assert len(rows) == 8
    assert all(r["rendered"] and r["audit"]["length"] >= 20 for r in rows)
    assert len({tuple(r["path"]) for r in rows}) == 8

def test_two_pointer_and_sha_audit_are_independent():
    for r in lane.main()["candidates"]:
        t = lane.norm(r["rendered"])
        a = r["audit"]
        assert a["two_pointer"] == all(t[i] == t[-1-i] for i in range(len(t)//2))
        assert a["forward_sha256"] == hashlib.sha256(t.encode()).hexdigest()
        assert a["reverse_sha256"] == hashlib.sha256(t[::-1].encode()).hexdigest()

def test_novelty_preflight_forbids_shortcuts():
    p = lane.main()
    assert all(p["novelty_preflight"].values())
    assert p["next_construction"].startswith("Add an independent")
