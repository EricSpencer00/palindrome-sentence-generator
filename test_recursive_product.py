import json
from pathlib import Path

from experiments.open_residual_cycle_certificate_20260922 import run as certificate
from llm_palindrome.recursive_product import Edge, materialize_pump, search, tape


def test_pumpable_nonempty_residual_cycle():
    # Prefix reaches debt L:"b".  The two loop edges change it to R:"c"
    # and back to L:"b" without ever closing; the two exit edges then close
    # only when both grammars reach F.
    left = {
        "S": (Edge("P", "ab", "A"),),
        "P": (Edge("P", "cb", "B"), Edge("F", "x", "A'")),
    }
    right = {
        "S": (Edge("Q", "a", "A"),),
        "Q": (Edge("Q", "cb", "B'"), Edge("F", "xb", "A'")),
    }
    r = search(left, right, reject_intermediate_closure=True)
    assert r.witnesses
    assert r.pumpable_cycles
    lengths = []
    for pump in r.pumpable_cycles:
        assert pump.states[0] == pump.states[-1]
        assert all(s.residual for s in pump.states)
        for repetitions in range(4):
            witness = materialize_pump(pump, repetitions)
            rendered = " ".join(witness.left_words + witness.right_words)
            letters = tape(rendered)
            assert letters == letters[::-1]
            lengths.append(len(letters))
    assert len(set(lengths)) >= 4


def test_reachable_but_not_coaccessible_cycle_is_not_pumpable():
    left = {"S": (Edge("S", "a"),)}
    right = {"S": (Edge("S", "aa"),)}
    r = search(left, right, reject_intermediate_closure=False)
    assert r.reachable
    assert not r.pumpable_cycles


def test_boundary_shift_acceptance():
    left = {"S": (Edge("L1", "ab"),), "L1": (Edge("F", "c"),)}
    # Right edges are selected outside-in; normal reading order is "cb a".
    right = {"S": (Edge("R1", "a"),), "R1": (Edge("F", "cb"),)}
    r = search(left, right)
    assert len(r.witnesses) == 1
    witness = r.witnesses[0]
    assert witness.left_words == ("ab", "c")
    assert witness.right_words == ("cb", "a")
    assert witness.left_boundaries == (2,)
    assert witness.reflected_right_boundaries == (1,)
    assert witness.left_phases == ("", "")
    assert witness.right_phases == ("", "")
    assert witness.left_roles == ("", "")
    assert witness.right_roles == ("", "")


def test_intermediate_empty_closure_is_counted_and_rejected():
    left = {"S": (Edge("S", "a"), Edge("F", "a"))}
    right = {"S": (Edge("S", "a"), Edge("F", "a"))}
    r = search(left, right)
    assert r.intermediate_empty_closures > 0
    # The direct S->F closure is valid; only the smaller loop closures are
    # rejected as reusable inner palindromes.
    assert len(r.witnesses) == 1
    assert not r.pumpable_cycles


def test_checked_in_cycle_certificate_matches_recomputed_audits():
    generated = certificate()
    stored = json.loads((Path(__file__).parent / "runs" /
                         "open-residual-cycle-certificate-20260922.json").read_text())
    assert generated["status"] == stored["status"]
    assert generated["cycle_states"] == stored["cycle_states"]
    assert [row["audit"]["letters"] for row in generated["rows"]] == [
        row["letters"] for row in stored["rows"]
    ]
    assert [row["audit"]["sha256_forward"] for row in generated["rows"]] == [
        row["sha256"] for row in stored["rows"]
    ]
