import json
from pathlib import Path
from experiments.exact_palindrome_graph_product_20260917 import CharacterGraph, exact_audit, run, solve_product, SIGNATURE

def test_fixture_oracle_and_live_rejection():
    result = run()
    assert result["fixture"]["rendered"] == ["live on time emit no evil"]
    assert result["fixture"]["result"]["rejected_unequal_edge_pairs"] > 0
    assert result["fixture"]["result"]["completions"]
    assert result["fixture"]["pairs"] == [{"left_path": "live on time", "right_path": "emit no evil", "full_tape": "live on time emit no evil"}]

def test_half_tape_is_not_exact_admission():
    assert not exact_audit("live on time")["exact"]

def test_lexical_lane_has_no_fabricated_duplicate_paths():
    lexical = run()["lexical_search"]["completed_paths"]
    assert all(row["left_path"] != row["right_path"] for row in lexical)
    assert all(" " in row["left_path"] and " " in row["right_path"] for row in lexical)

def test_bounded_lexical_graph_can_represent_three_word_path():
    graph = CharacterGraph.from_phrases(["one two three"], "bounded")
    assert len(graph.accepting_paths) == 1
    assert graph.accepting_paths[next(iter(graph.accepting_paths))].count(" ") == 2

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
