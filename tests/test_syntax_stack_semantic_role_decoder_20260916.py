import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_syntax_stack_keeps_complete_prose_and_zero_exact_closures():
    run = json.loads((ROOT / "runs" / "syntax-stack-semantic-role-decoder-20260916.json").read_text())
    assert run["states_examined"] == 81
    assert run["exact_count"] == 0
    assert run["mechanically_admitted_count"] == 0
    assert run["provenance"]["catalogue_or_corpus_import"] is False
    assert all(row["provenance"]["human_authored_frames"] for row in run["failed_attempts"])
    assert all(row["next_repair_operator"] for row in run["failed_attempts"])


def test_syntax_stack_best_surface_is_not_a_palindrome():
    run = json.loads((ROOT / "runs" / "syntax-stack-semantic-role-decoder-20260916.json").read_text())
    text = run["best_rendered_candidates"][0]["rendered"]
    tape = "".join(re.findall(r"[A-Za-z]", text)).lower()
    assert len(tape) >= 60
    assert tape != tape[::-1]
