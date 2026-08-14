"""The verifiable half of the reward.

The point of these properties is that a policy cannot trade them away, so the
tests are mostly about `verify` refusing rather than scoring.
"""
from __future__ import annotations

import pytest

from llm_palindrome.tunable import DEFAULT, TunableScorer
from llm_palindrome.verify import InvariantViolation, verify

PALINDROME = ["rats", "live", "on", "no", "evil", "star"]


def test_a_real_palindrome_verifies():
    v = verify(PALINDROME)
    assert v.is_palindrome and v.closed
    assert v.letters == 20
    assert v.words == 6
    assert v.adjacent_repeats == 0


def test_a_non_palindrome_raises_rather_than_scoring_low():
    with pytest.raises(InvariantViolation):
        verify(["rats", "live", "on", "elephant"])


def test_words_outside_the_vocabulary_raise():
    with pytest.raises(InvariantViolation):
        verify(PALINDROME, vocabulary={"rats", "live", "on"})


def test_strict_off_reports_instead_of_raising():
    v = verify(["rats", "live", "on", "elephant"], strict=False)
    assert not v.is_palindrome


def test_length_reward_saturates_at_the_target():
    short = verify(["step", "on", "no", "pets"])
    assert short.reward(target_letters=1000) < short.reward(target_letters=12)
    # Past the target, extra letters stop paying.
    assert short.reward(target_letters=6) == short.reward(target_letters=12)


def test_adjacent_repeats_cost_reward():
    clean = verify(["step", "on", "no", "pets"])
    repeated = verify(["step", "on", "on", "no", "no", "pets"])
    assert repeated.adjacent_repeats == 2
    assert repeated.reward(20) < clean.reward(20)


def test_failure_to_close_is_the_worst_outcome():
    assert verify([]).reward(200) == -10.0


def test_tunable_reproduces_its_default_feature_set():
    s = TunableScorer(DEFAULT)
    f = s.features(("rats", "live"), (), "L", "live", "append")
    assert len(f) == len(DEFAULT)
    assert f[1] == 4.0                      # length of "live"
    assert f[2] == 0.0                      # first use
    assert f[3] == 0.0                      # neighbour is "rats", not a repeat


def test_tunable_sees_an_adjacent_repeat_in_both_growth_directions():
    s = TunableScorer(DEFAULT)
    appended = s.features(("on", "on"), (), "L", "on", "append")
    prepended = s.features((), ("on", "on"), "R", "on", "prepend")
    assert appended[3] == 1.0
    assert prepended[3] == 1.0
