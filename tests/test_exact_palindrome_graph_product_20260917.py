import json
from pathlib import Path
from experiments.exact_palindrome_graph_product_20260917 import CharacterGraph, exact_audit, run, solve_product, SIGNATURE

def test_fixture_oracle_and_live_rejection():
    result = run()
    assert result["fixture"]["rendered"] == ["live on time emit no evil"]
    assert result["fixture"]["result"]["rejected_unequal_edge_pairs"] > 0
    assert result["fixture"]["result"]["completions"]

def test_root_intersections_are_reported_without_sentence_candidates():
    result = run()
    assert set(result["template_domains"]) == {"declarative", "question", "imperative"}
    assert result["root_character_intersections"]
    assert "completed paths only" in result["lexical_search"]["grammar_gate"]

def test_graph_has_boundary_provenance_and_backpointer():
    graph = CharacterGraph.from_words(["ab"], "test")
    assert any(e.char is None and "word-boundary" in e.provenance for edges in graph.edges.values() for e in edges)
    assert solve_product(graph, graph)["completions"][0]["backpointer"]

def test_run_artifact_matches_signature():
    run()
    artifact = json.loads(Path("runs/exact-palindrome-graph-product-20260917.json").read_text())
    assert artifact["signature"] == SIGNATURE
    assert artifact["audits"]["anti_shortcut_checks"]
