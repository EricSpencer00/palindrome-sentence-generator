import json, re
from pathlib import Path

ROOT=Path(__file__).parents[1]
def letters(s): return "".join(re.findall(r"[a-z]",s.lower()))

def test_repair_is_bounded_debt_directed_and_complete():
    run=json.loads((ROOT/"runs/dependency-mirror-pair-repair-20260916.json").read_text())
    assert run["novelty_preflight"]["passed"]
    assert run["repair_count"]==run["base_count"]==4
    assert run["exact_count"]==0
    assert all(r["rendered"].endswith(".") and r["paired_rendered"].endswith(".") for r in run["candidates"])
    assert max(r["letters"] for r in run["candidates"])>=60
    assert all(r["selection"]["trials"]==9 for r in run["candidates"])
    assert all(r["provenance"]["word_order_mirrored"] is False for r in run["candidates"])
    assert all(r["repair"]["operator"].startswith("global-reverse-tape") for r in run["candidates"])
