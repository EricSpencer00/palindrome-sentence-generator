from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("luna_cfg_frame_repair_20260917", ROOT / "experiments/luna_cfg_frame_repair_20260917.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fresh_frame_preflight_and_chart():
    result = MODULE.run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["grammar"]["two_ended"] is True
    assert result["best"]["chart_states"]
    assert result["best"]["provenance"]["parent_exact_terminals_reused"] is False


def test_failure_is_complete_and_fail_closed():
    row = MODULE.run()["best"]
    assert row["audit"]["letters"] > 38
    assert row["audit"]["independent_two_pointer_exact"] is False
    assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
    assert row["mechanically_admitted"] is False
    assert row["anti_shortcut_flags"]["word_order_mirror"] is False
    assert row["provenance"]["posthoc_resegmentation"] is False
