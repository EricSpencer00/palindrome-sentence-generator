import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_phrase_equation_inventory_preserves_complete_probes_and_zero_closure():
    run = json.loads((ROOT / "runs" / "phrase-equation-inventory-solver-20260916.json").read_text())
    assert run["candidate_count"] == 243
    assert run["closure_count"] == 0
    assert run["reader_eligible"] is False
    assert all(row["provenance"]["catalogue_text_used"] is False for row in run["candidates"])
    assert all(row["audit"]["two_pointer"]["exact"] is False for row in run["candidates"])


def test_phrase_equation_best_surface_is_long_and_nonexact():
    run = json.loads((ROOT / "runs" / "phrase-equation-inventory-solver-20260916.json").read_text())
    text = run["best_actual_prose"]["rendered"]
    tape = "".join(re.findall(r"[A-Za-z]", text)).lower()
    assert len(tape) >= 60
    assert tape != tape[::-1]
