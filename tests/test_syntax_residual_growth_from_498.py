import json
from pathlib import Path
from llm_palindrome.validator import is_palindrome

def test_saved_syntax_children_are_independently_exact():
    p=json.loads(Path("runs/syntax-residual-growth-from-498-20261001.json").read_text())
    assert p["parent_letters"] == 498
    for row in p["rows"]:
        assert row["normalized_length"] > p["parent_letters"]
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal"]
        assert is_palindrome(row["rendered"])
