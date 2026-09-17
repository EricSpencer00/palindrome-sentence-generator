import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "semantic_slot_attachment_repair",
    ROOT / "experiments/semantic_slot_attachment_repair_20260916_luna.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_paired_semantic_edit_records_live_obligation_and_dual_audit():
    run = module.build_run()
    repair = run["scene"]["repair"]
    assert run["novelty_preflight"]["passed"]
    assert run["novelty_preflight"]["duplicate_sweep_run"] is False
    assert repair["changed_slot_count"] == 1
    assert repair["changed_attachment_count"] == 1
    assert repair["character_obligation"]
    assert run["scene"]["independent_audits_agree"]
    assert run["scene"]["repaired"]["prose_shape"]["complete_scene"]
    assert run["scene"]["repaired"]["anti_shortcut"]["reverse_or_wrapper_rendering"] is False


def test_run_json_is_unique_and_has_provenance():
    path = ROOT / "runs/semantic-slot-attachment-repair-20260916-luna.json"
    assert path.exists()
    run = json.loads(path.read_text())
    assert run["experiment"] == module.EXPERIMENT
    assert run["provenance"]["generator_sha256"]
    assert run["provenance"]["registry_sha256"]
    assert run["scene"]["repaired"]["sha256"]["forward"]
    assert run["scene"]["repaired"]["sha256"]["reverse"]
