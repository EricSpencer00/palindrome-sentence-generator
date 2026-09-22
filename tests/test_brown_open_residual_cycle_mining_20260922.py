from experiments.brown_open_residual_cycle_mining_20260922 import mine_cycles


def row(text, role="np", count=1):
    return {"text": text, "tags": ["NOUN"], "role": role,
            "count": count, "source": "synthetic test"}


def test_miner_recovers_cyclic_reversal_equation_at_nonempty_debt():
    # Right tape dcba exposes abcd = r+s. Left tape cdab is s+r.
    cycles, stats = mine_cycles({
        "cdab": [row("left item")],
        "dcba": [row("right form")],
    }, max_residual=2, max_rows=20)
    assert stats["role_compatible_disjoint_cycles"] == 2
    cycle = next(row for row in cycles if row["residual"] == "ab")
    assert cycle["equation"]["exact_open_cycle"] is True


def test_miner_requires_compatible_roles_and_disjoint_words():
    cycles, stats = mine_cycles({
        "cdab": [row("left token", role="np")],
        "dcba": [row("right token", role="vp_or_adjunct")],
    }, max_residual=2, max_rows=20)
    assert cycles == []
    assert stats["role_compatible_disjoint_cycles"] == 0


def test_targeted_residuals_do_not_expand_unrelated_rotations():
    index = {
        "cdab": [row("left item")],
        "dcba": [row("right form")],
    }
    cycles, stats = mine_cycles(index, max_residual=4, max_rows=20,
                                target_residuals=("ab",))
    assert [cycle["residual"] for cycle in cycles] == ["ab"]
    assert stats["rotations_tested"] == 1
    assert stats["target_residuals"] == ["ab"]
