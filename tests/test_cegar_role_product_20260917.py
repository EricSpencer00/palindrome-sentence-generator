from itertools import product as cartesian

from experiments.cegar_role_product_20260917 import (
    _audit,
    compile_pattern,
    product,
    withheld_seed_witness,
)


def test_withheld_seed_is_recovered_by_live_product_but_not_promoted():
    witness = withheld_seed_witness()
    assert witness["withheld"] is True
    assert witness["not_a_generated_candidate"] is True
    assert witness["letters"] == 38
    assert witness["product_records"] == 1
    assert witness["audit"]["exact"] is True
    assert witness["audit"]["forward_sha256"] == witness["audit"]["reverse_sha256"]


def test_product_matches_independent_small_grammar_oracle():
    roles = ("left", "right", "center")
    banks = {
        "left": ("a", "ab", "ba"),
        "right": ("b", "aa", "aba"),
        "center": ("a", "bb"),
    }
    grammar = compile_pattern(roles, banks)
    result = product(grammar)
    expected = set()
    for words in cartesian(*(banks[role] for role in roles)):
        tape = "".join(words)
        if tape == tape[::-1]:
            expected.add(words)
    observed = {tuple(row["words"]) for row in result["records"]}
    assert observed == expected
    for words in observed:
        assert _audit(" ".join(words))["exact"] is True


def test_product_has_no_finished_tape_reverse_step():
    # The only exact path is assembled through live matching edges; a
    # post-hoc reverse of a completed sentence is not part of the interface.
    grammar = compile_pattern(("left", "right"), {
        "left": ("ab",),
        "right": ("ba",),
    })
    result = product(grammar)
    assert [row["words"] for row in result["records"]] == [["ab", "ba"]]
    assert "reverse" not in result
