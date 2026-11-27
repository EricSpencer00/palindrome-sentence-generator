"""Question marks: a third free punctuation token that ends a sentence like a
full stop, chosen only where the prose model has seen questions. No corpora
needed: the model is trained on a fake Brown and Gutenberg."""
import collections
import sys
import types

import pytest

from llm_palindrome import paragraph as P
from llm_palindrome.validator import is_palindrome

BROWN = [
    ["Did", "you", "see", "a", "cat", "?"],
    ["Was", "it", "a", "cat", "I", "saw", "?"],
    ["I", "saw", "a", "cat", ",", "and", "a", "dog", "."],
    ["A", "cat", "was", "on", "a", "mat", "!"],
]
GUTENBERG = [
    ["Where", '?"', "said", "the", "cat", "."],
    ["Who", "was", "it", ",", "?--"],
]


@pytest.fixture
def lm(monkeypatch):
    def corpus(sents):
        return types.SimpleNamespace(sents=lambda: [list(s) for s in sents])
    nltk = types.ModuleType("nltk")
    nltk.corpus = types.ModuleType("nltk.corpus")
    nltk.corpus.brown, nltk.corpus.gutenberg = corpus(BROWN * 3), corpus(GUTENBERG * 3)
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", nltk.corpus)
    from llm_palindrome.prose_lm import ProseLM, VERSION
    m = ProseLM()
    assert m.version == VERSION
    return m


def test_question_mark_is_a_free_sentence_end():
    assert "?" in P.PUNCT and "?" in P.ENDS and "?" in P.MARKS
    assert P.letters("?") == ""


def test_render_capitalises_after_a_question():
    assert P.render(("did", "i", "?", "so", "i", "did")) == "Did I? So I did."
    assert P.render(("was", "it", "a", "cat", "i", "saw", "?")) == "Was it a cat I saw?"
    assert P.render(("no", ",", "i", "did", ".", "so")) == "No, I did. So."


def test_a_palindrome_with_a_question_validates():
    assert is_palindrome(P.render(("was", "it", "a", "cat", "i", "saw", "?")))
    assert is_palindrome(P.render(("did", "i", "?", "did")) + " ")


def test_tokenisation_keeps_questions_and_reads_bang_as_stop(lm):
    assert lm.uni["?"] > 0
    assert "!" not in lm.uni
    # Gutenberg's '?"' and '?--' are questions too; a comma before one is dropped.
    assert lm.tri.get(("was", "it", "?"))
    # The sentence after a question starts after a "?".
    assert lm.bi.get(("?", "i"))
    # A question opener prefers "?", a statement prefers ".".
    assert lm.fwd("cat", "i", "saw") > lm.fwd("cat", "i", "did")
    assert lm.fwd("a", "cat", "?") > lm.fwd("a", "cat", ".")
    assert lm.fwd("a", "dog", ".") > lm.fwd("a", "dog", "?")


def test_closures_stay_exact_and_can_ask(lm):
    lex = P.Lexicon.__new__(P.Lexicon)
    lex.words = {w for w in lm.uni if w.isalpha()}
    lex.freq = {w: lm.uni[w] for w in lex.words}
    lex.prefix, lex.suffix = collections.defaultdict(list), collections.defaultdict(list)
    for w in lex.words:
        for k in range(1, len(w)):
            lex.prefix[w[:k]].append(w)
            lex.suffix[w[-k:]].append(w)
    lex.before = collections.defaultdict(list)
    for (a, b) in lm.bi:
        if a in lex.words:
            lex.before[b].append(a)
    closed = P.grow(lm, lex, 12, beam=60, centre=("i",), seed=1)
    assert closed
    texts = [P.render(s.seq) for s in closed]
    assert all(is_palindrome(t) for t in texts)
    assert all(t.endswith((".", "?")) for t in texts)
    for s in closed:
        assert s.seq[0] not in P.MARKS
        assert all(not (a in P.MARKS and b in P.MARKS) for a, b in zip(s.seq, s.seq[1:]))
