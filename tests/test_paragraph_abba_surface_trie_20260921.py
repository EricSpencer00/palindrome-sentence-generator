import json
from pathlib import Path
import experiments.paragraph_abba_surface_trie_20260921 as m

def test_joint_surface_lane_has_distinct_complete_roles_and_audit():
    d = m.run()
    assert d["stats"]["candidates"] == 24
    assert d["stats"]["exact_candidates"] == 0
    assert d["stats"]["max_live_support"] == 0
    for row in d["rendered_candidates"]:
        assert len(set(row["units"])) == 4
        assert row["topology"] == ["A1", "B1", "B2", "A2"]
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert row["provenance"]["joint_decoding"]
        assert not row["gates"]["exact"]

def test_artifact_is_reproducible_and_shortcut_gated():
    d = m.run(); p = Path(m.OUT); p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(d, indent=2) + "\n")
    saved = json.loads(p.read_text())
    assert saved["stats"] == d["stats"]
    assert d["novelty_preflight"]["status"] == "passed"
    assert d["novelty_preflight"]["fixed_bank_sweep"] is False
