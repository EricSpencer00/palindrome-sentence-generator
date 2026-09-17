import json, runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
M=runpy.run_path(str(ROOT/"experiments/phrase_pair_semantic_graph_20260917.py"))

def test_graph_edges_close_and_phrases_are_not_self_palindromes():
    for p in M["BANK"]:
        assert M["tape"](p["left"]) != M["tape"](p["left"])[::-1]
        assert M["tape"](p["right"]) != M["tape"](p["right"])[::-1]
    assert all(e["exact"] for e in [M["solve"](p) for p in M["BANK"]])

def test_run_requires_independent_exact_audits():
    out=M["run"]()
    assert out["novelty_preflight"]["passed"]
    assert out["mechanically_admitted"]
    assert out["exact_audit"]["independent_agreement"]
    assert out["rendered"].endswith("no pets.")
