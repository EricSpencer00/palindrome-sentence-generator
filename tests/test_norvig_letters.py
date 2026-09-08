"""Integration controls for the locally cached, unmodified Norvig reference."""
from collections import Counter
from pathlib import Path
import re
import pytest

from experiments.norvig_letters import ROOT, run

pytestmark = pytest.mark.skipif(
    not (ROOT/'runs/norvig/pal3.py').exists(), reason='Norvig reference source not cached')


@pytest.mark.parametrize('feasible', [False, True])
def test_dynamic_inventory_restores_counts_on_backtracking(tmp_path, feasible):
    state = run(0, True, tmp_path, feasible=feasible)
    prefixes = dict(state.dict.prefixes)
    suffixes = dict(state.dict.suffixes)
    words = state.words.copy()
    active = getattr(state, 'active', set()).copy()
    # First valid branch in the original worked letter-by-letter example.
    for action in ['r', 'e', ',']:
        state.do(action)
    assert state.total == 26
    assert 'acare' in state.set
    if feasible:
        assert 'acare' not in state.active
    for action in [',', 'e', 'r']:
        state.undo(action)
    assert state.total == 21 and state.L == 'aca' and state.R == ''
    assert dict(state.dict.prefixes) == prefixes
    assert dict(state.dict.suffixes) == suffixes
    assert +state.words == +words
    if feasible:
        assert state.active == active
    text = (tmp_path/'palindrome.txt').read_text()
    letters = re.sub('[^a-z]', '', text.lower())
    assert letters == letters[::-1]


def test_used_and_word_capped_phrases_are_not_available(tmp_path):
    state = run(0, True, tmp_path, feasible=True)
    assert not state.inventory_allowed('aman')
    assert state.inventory_allowed('acare')
    state.words['care'] = 3
    state.refresh('acare')
    assert not state.inventory_allowed('acare')
    assert 'acare' not in state.active
    state.words['care'] = 0
    state.refresh('acare')
    assert 'acare' in state.active


def test_resume_keeps_the_verified_seed(tmp_path):
    first = tmp_path/'first'
    second = tmp_path/'second'
    run(0, True, first)
    resumed = run(0, True, second, resume=first)
    assert resumed.best_letters == 21
    assert (first/'palindrome.txt').read_bytes() == (second/'palindrome.txt').read_bytes()


def test_independent_audit_rejects_letter_corruption(tmp_path):
    from experiments.audit_norvig_result import audit
    run(0, True, tmp_path)
    dictionary = ROOT/'runs/norvig/npdict.txt'
    reference = ROOT/'runs/norvig/pal21txt.html'
    result = audit(tmp_path, dictionary, reference)
    assert result['reference_letters'] == 90439
    assert result['all_phrases_in_original_dictionary']
    path = tmp_path/'palindrome.txt'
    path.write_text(path.read_text().replace('man', 'men', 1))
    with pytest.raises(AssertionError):
        audit(tmp_path, dictionary, reference)
