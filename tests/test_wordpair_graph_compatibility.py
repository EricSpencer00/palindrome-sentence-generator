from itertools import product

from llm_palindrome.wordpair_graph import extendable


def _strings(alphabet: str, max_length: int) -> list[str]:
    return [
        "".join(chars)
        for length in range(max_length + 1)
        for chars in product(alphabet, repeat=length)
    ]


def _can_close_with_free_character_continuations(left: str, right: str) -> bool:
    """Oracle for whether unconstrained future edges can reconcile both tapes."""
    continuations = _strings("ab", 2)
    exposed_right = right[::-1]
    return any(
        left + left_extra == exposed_right + right_extra
        for left_extra in continuations
        for right_extra in continuations
    )


def test_extendable_matches_exhaustive_character_continuation_oracle():
    for left in _strings("ab", 2):
        for right in _strings("ab", 2):
            assert extendable(left, right) == _can_close_with_free_character_continuations(
                left, right
            ), (left, right)


def test_extendable_preserves_residual_debt_but_rejects_fixed_conflicts():
    assert extendable("ab", "a")  # the unmatched b can be discharged inward
    assert extendable("a", "ba")  # reverse(right) starts with a; residual b is live
    assert extendable("ab", "ba")  # complete overlap
    assert not extendable("ab", "xy")  # equal-length tapes already conflict
    assert not extendable("ab", "z")
