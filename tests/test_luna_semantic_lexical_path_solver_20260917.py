import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/luna_semantic_lexical_path_solver_20260917.py"
RUN = ROOT / "runs/luna-semantic-lexical-path-solver-20260917.json"


def test_semantic_path_run_is_reproducible_and_audited():
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
    data = json.loads(RUN.read_text())
    assert data["signature"].startswith("semantic-lexical-path")
    assert data["stats"]["semantic_paths"] == 1
    assert data["stats"]["lexical_realizations"] == 32
    assert data["novelty_preflight"]["pos_sweep"] is False
    assert data["novelty_preflight"]["scene_lattice"] is False
    repaired = data["repaired_candidate"]
    assert repaired["repaired_role"] == "event"
    assert repaired["held_out_synonym"] == "points out"
    assert repaired["audit"]["rendered"].startswith("The young botanist points out")
    assert len(repaired["edge_obligations"]) == 4
    second = data["second_repair"]
    assert second["repaired_role"] == "locative"
    assert second["held_out_synonym"] == "near the market"
    assert second["audit"]["rendered"].startswith("The young botanist points out a coastal inlet near the market")
    assert second["edge_obligations"][0]["satisfied"] is True
    assert second["edge_obligations"][1]["satisfied"] is True
    assert second["edge_obligations"][2]["satisfied"] is True
    third = data["third_repair"]
    assert third["repaired_role"] == "consequence"
    assert third["held_out_synonym"] == "and keeps a quiet secret"
    assert third["audit"]["rendered"].endswith("and keeps a quiet secret.")
    assert all(item["satisfied"] for item in third["edge_obligations"])
    expanded = data["expanded_consequence_candidates"]
    assert len(expanded) == 2
    assert all(all(item["satisfied"] for item in row["edge_obligations"]) for row in expanded)
    assert all(not row["audit"]["exact"] for row in expanded)
    assert data["candidate_scene"] != third["audit"]["rendered"]
    joint = data["joint_agent_consequence_candidates"]
    assert len(joint) == 3
    assert all(all(item["satisfied"] for item in row["edge_obligations"]) for row in joint)
    assert all(not row["audit"]["exact"] for row in joint)
    assert all(row["audit"]["mechanical_checks"]["lexicon_words"] for row in joint)
    paired = data["joint_theme_locative_candidates"]
    assert len(paired) == 3
    assert all(all(item["satisfied"] for item in row["edge_obligations"]) for row in paired)
    assert all(not row["audit"]["exact"] for row in paired)
    assert all(row["audit"]["mechanical_checks"]["distinct_words"] for row in paired)
    event_theme = data["joint_event_theme_candidates"]
    assert len(event_theme) == 3
    assert all(all(item["satisfied"] for item in row["edge_obligations"]) for row in event_theme)
    assert all(not row["audit"]["exact"] for row in event_theme)
    assert all(row["audit"]["mechanical_checks"]["lexicon_words"] for row in event_theme)
    for row in data["candidates"]:
        assert row["audit"]["normalized_sha256"] != row["audit"]["reverse_sha256"] or row["audit"]["exact"]
        assert len(row["edge_obligations"]) == 4


def test_candidate_is_real_rendered_prose():
    data = json.loads(RUN.read_text())
    assert data["candidate_scene"].endswith(".")
    assert "the" in data["candidate_scene"].lower()
    assert data["provenance"]["generator_sha256"]
