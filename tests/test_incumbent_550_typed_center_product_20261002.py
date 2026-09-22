from experiments.incumbent_550_typed_center_product_20261002 import (
    WINNER_SHA256,
    build_payload,
    run_product,
)


def test_typed_product_reproduces_live_state_counts() -> None:
    counts, accepted = run_product()

    assert counts == [1, 1, 9, 45, 405, 54]
    assert len(accepted) == 54


def test_typed_product_saves_all_unique_exact_children() -> None:
    payload = build_payload()

    assert payload["stats"]["independently_exact_children"] == 54
    assert payload["stats"]["unique_centers"] == 54
    assert len({row["center"] for row in payload["rows"]}) == 54
    assert all(row["audit"]["two_pointer_exact"] for row in payload["rows"])
    assert all(row["audit"]["project_validator_exact"] for row in payload["rows"])


def test_typed_product_winner_is_deterministic() -> None:
    payload = build_payload()
    winner = next(row for row in payload["rows"] if row["id"] == payload["active_method_winner"])

    assert winner["center"] == "Leon, Aidan stops a rat. Tara spots Nadia, Noel."
    assert winner["audit"]["letters"] == 558
    assert winner["audit"]["sha256_forward"] == WINNER_SHA256
    assert winner["residual_trace"][-2]["owner"] == "R"
    assert winner["residual_trace"][-2]["residual"] == "rat"
    assert winner["residual_trace"][-1]["owner"] == "-"
