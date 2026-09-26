import time

from llm_palindrome.search import State, WordTries, beam_search
from llm_palindrome.centerout import COState, centerout_search
from llm_palindrome.admission import mechanical_admission_checks
from experiments.paired_directional_search_benchmark_20260925 import (
    STRUCTURAL_CHECKS,
    render,
    state_gate,
)


class ZeroScorer:
    def word_delta(self, left, right, placement, word, growth):
        return 0.0


class SlowEmptyTries(WordTries):
    def left_candidates(self, overhang, limit=200):
        time.sleep(0.03)
        return []

    def right_candidates(self, overhang, limit=200):
        time.sleep(0.03)
        return []


def test_beam_search_can_continue_from_exact_compatible_endpoint_state():
    state = State(0.0, ("some",), ("memos",), "m", "L", 0.0)
    result = beam_search(
        WordTries(["some", "memos", "a", "am", "man"]),
        ZeroScorer(),
        min_letters=9,
        beam_width=8,
        max_steps=2,
        candidate_limit=32,
        initial_state=state,
    )
    assert result[:1] == ["some"]
    assert result[-1:] == ["memos"]


def test_centerout_accepts_a_spaced_palindromic_seed_phrase():
    words = centerout_search(
        WordTries(["a", "am", "madam"]),
        ZeroScorer(),
        center="Some men interpret nine memos",
        min_letters=25,
        beam_width=4,
        max_steps=1,
        candidate_limit=16,
    )
    assert "some men interpret nine memos" in " ".join(words).lower()


def test_outside_in_closure_gate_runs_only_for_palindromic_debt():
    seen = []
    state = State(0.0, ("a",), ("a",), "ab", "L", 0.0)
    beam_search(
        WordTries([]),
        ZeroScorer(),
        min_letters=1,
        max_steps=1,
        initial_state=state,
        allow_closed=lambda left, right: seen.append((left, right)) or False,
    )
    assert seen == []


def test_outside_in_returns_best_closure_when_deadline_stops_expansion():
    state = State(0.0, ("a",), ("a",), "", "L", 0.0)
    words = beam_search(
        SlowEmptyTries([]), ZeroScorer(), min_letters=2, max_steps=2,
        initial_state=state, allow_closed=lambda left, right: True,
        deadline=time.monotonic() + 0.01,
    )
    assert words == ["a", "a"]


def test_centerout_returns_best_closure_when_deadline_stops_expansion():
    state = COState(0.0, ("a",), ("a",), "", "R", 0.0)
    words = centerout_search(
        SlowEmptyTries([]), ZeroScorer(), min_letters=2, max_steps=2,
        initial_state=state, allow_closed=lambda left, right: True,
        deadline=time.monotonic() + 0.01,
    )
    assert words == ["a", "a"]


def test_both_searches_reach_known_target_and_fire_native_closure_gate():
    tries = WordTries(["level"])
    outside_seen = []

    def admits(left, right, seen):
        words = tuple(left) + tuple(right)
        text = render(words)
        checks = mechanical_admission_checks(text, min_letters=10, max_letters=10)
        seen.append(checks)
        return all(checks[name] for name in STRUCTURAL_CHECKS)

    outside = beam_search(
        tries, ZeroScorer(), min_letters=10, beam_width=4, max_steps=4,
        candidate_limit=4, per_parent=4, seed=1, diversity=0.0,
        allow_state=lambda left, right: state_gate(tuple(left) + tuple(right), 10),
        allow_closed=lambda left, right: admits(left, right, outside_seen),
    )
    center_seen = []
    center = centerout_search(
        tries, ZeroScorer(), min_letters=10, beam_width=4, max_steps=4,
        candidate_limit=4, per_parent=4, seed=1, diversity=0.0,
        allow_state=lambda left, right: state_gate(tuple(left) + tuple(right), 10),
        allow_closed=lambda left, right: admits(left, right, center_seen),
    )

    assert outside and "".join(outside).replace(" ", "") == "".join(outside).replace(" ", "")[::-1]
    assert center and "".join(center).replace(" ", "") == "".join(center).replace(" ", "")[::-1]
    assert outside_seen and all(
        checks[name] for checks in outside_seen for name in STRUCTURAL_CHECKS
    )
    assert center_seen and all(
        checks[name] for checks in center_seen for name in STRUCTURAL_CHECKS
    )
