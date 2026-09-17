import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def tape(s):
    return re.sub(r"[^a-z]", "", s.casefold())


def test_joint_terminal_boundary_lane_keeps_controls_and_independent_audits():
    run = json.loads((ROOT / "runs/luna-joint-terminal-boundary-scene-repair-20260917.json").read_text())
    assert run["novelty_preflight"]["passed"]
    assert run["stats"] == {"scene_families": 3, "lattice_states": 24, "joint_repairs": 3, "rendered": 27, "exact": 0, "mechanically_admitted": 0, "longest_letters": 93}
    assert len(run["controls"]) == 3
    for row in run["rows"]:
        t = tape(row["rendered"])
        p = row["exact_audit"]["pointer"]
        s = row["exact_audit"]["sha"]
        assert p["letters"] == len(t)
        assert s["forward"] == hashlib.sha256(t.encode()).hexdigest()
        assert s["reverse"] == hashlib.sha256(t[::-1].encode()).hexdigest()
        assert p["exact"] == s["exact"]
        assert row["rendered"].endswith(".")
        assert row["provenance"]["fresh_authored_scene"]
        assert not row["provenance"]["catalogue_imported"]
        assert not row["provenance"]["reversed_finished_sentence"]
        assert not row["provenance"]["word_order_mirror"]
        assert not row["provenance"]["repeated_unit"]
        assert row["next_repair"]
    assert all(row["phase"] == "joint_repair" for row in run["controls"])
