import hashlib
import json
import re

from experiments.norvig_long import load_phrases, render, search


def test_seed_and_render_keep_exact_letters():
    text = render(['a man','a plan'], ['panama','a canal'], paragraph_words=3)
    normalized = re.sub('[^a-z]', '', text.lower())
    assert normalized == normalized[::-1]
    assert '\n\n' in text
    assert text.startswith('A man, a plan')
    assert text.endswith('panama.')


def test_dictionary_deduplicates_letter_identity(tmp_path):
    path = tmp_path/'dict.txt'
    path.write_text('A cat\nAc at\nDo do\n!!!\nA dog\n')
    assert load_phrases(path) == ['a cat', 'a dog']


def test_search_grows_known_seed_and_checks_saved_artifact(tmp_path):
    # The worked example in Norvig's algorithm, with dead ends for backtracking.
    result = search(['a caddy', 'roydd', 'ore', 'belize'], 2, 0, tmp_path)
    text = (tmp_path/'palindrome.txt').read_text()
    normalized = re.sub('[^a-z]', '', text.lower())
    assert normalized == normalized[::-1]
    assert len(normalized) == result['letters']
    assert hashlib.sha256((tmp_path/'palindrome.txt').read_bytes()).hexdigest() == result['sha256']
    assert result['letters'] > 21
    units = json.loads((tmp_path/'phrases.json').read_text())
    assert len(units) == len(set(units))
    words = re.findall('[a-z]+', text.lower())
    assert all(a != b for a,b in zip(words,words[1:]))
