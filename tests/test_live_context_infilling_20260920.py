import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from live_context_infilling_20260920 import State, consume  # noqa: E402


def test_live_residual_accepts_unpaired_left_prefix_and_rejects_conflict():
    s = State(("A",), ("a",), "", "none", 1, ())
    good, why = consume(s, ("A", "careful"), ("a",))
    assert why is None and good.residual_side == "left"
    bad, why = consume(s, ("A", "careful"), ("a", "x"))
    assert bad is None and why == "character_conflict"


def test_artifact_is_bounded_and_fail_closed():
    subprocess.run([sys.executable, str(ROOT / "experiments/live_context_infilling_20260920.py")], check=True, cwd=ROOT)
    x = json.loads((ROOT / "runs/live-context-infilling-20260920.json").read_text())
    assert x["parameters"] == {"starts": 16, "beam": 32, "rounds": 12, "target_letters": [40, 80]}
    assert x["stats"]["final_live_states"] <= 32
    assert x["stats"]["finished_exact_gt38"] == 0
    assert all(s["right_final_category"] == "NOUN" and s["right_clause_final"] for s in x["final_live"])
    assert all((not s["complete_clause"]) or (s["right_words"][0] in {"A", "The"})
               for s in x["final_live"])
    assert x["novelty_preflight"]["finished_tape_reversal"] is False
    assert x["novelty_preflight"]["post_hoc_repair"] is False
