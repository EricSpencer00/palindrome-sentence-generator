"""Regression tests for the structural graph sampler."""
import importlib.util
from pathlib import Path
import random

from llm_palindrome.search import WordTries


def _graph_module():
    path = Path("experiments/graph_search.py")
    spec = importlib.util.spec_from_file_location("graph_search_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graph_accepts_the_same_palindromic_overhang_center_as_beam_search():
    graph_search = _graph_module()
    graph = graph_search.Graph(WordTries(["level"]), max_overhang=8)
    counts = graph.closure_counts(max_letters=5, max_units=1)
    assert counts[0, 5, 1] > 0
    assert graph.sample(counts, 5, 1, random.Random(0)) == ["level"]


def test_primitive_mode_stops_after_the_first_nonroot_closure():
    graph_search = _graph_module()
    graph = graph_search.Graph(WordTries(["level"]), max_overhang=8)
    counts = graph.closure_counts(max_letters=10, max_units=2, primitive=True)
    assert counts[0, 10, 2] == 0
