from itertools import combinations

from experiments.sentence_intersection import controls, intersect, letters, repetition_ok


def rows(texts):
    return [{'text': text, 'source': 'test'} for text in texts]


def test_controls_are_calibrated_and_separate():
    assert controls()['passed']
    assert intersect(rows(['The dog ran home']))['pairs'] == []


def test_exact_lookup_matches_brute_force_and_preserves_segmentations():
    texts = ['abc def', 'ab cdef', 'fed cba', 'fedc ba', 'other text',
             'Go hang a salami', "I'm a lasagna hog"]
    expected = {frozenset((a, b)) for a, b in combinations(texts, 2)
                if letters(a) == letters(b)[::-1] and letters(a) != letters(a)[::-1]
                and repetition_ok(a + ' ' + b)}
    actual = intersect(rows(texts))
    got = {frozenset((p['left']['text'], p['right']['text'])) for p in actual['pairs']}
    assert got == expected
    assert len(actual['pairs']) == 5


def test_self_palindromes_and_punctuation_duplicates_cannot_be_pairs():
    result = intersect(rows(['Never odd or even', 'never odd or even', '!!!']))
    assert len(result['centres']) == 1
    assert result['pairs'] == []
    assert result['repetition_rejected'] == 1


def test_repetition_is_checked_across_pair_seam():
    result = intersect(rows(['ab c', 'c ba']))
    assert result['pairs'] == []
    assert not repetition_ok('do do do')
    assert not repetition_ok('one two one two one two')
    assert repetition_ok('Items draw award')


def test_boundary_counts_are_necessary_not_sufficient():
    result = intersect(rows(['abc def', 'fed xyz']))
    assert result['reverse_prefix_compatible']['3'] == 1
    assert result['pairs'] == []
