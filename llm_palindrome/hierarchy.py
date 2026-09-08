"""Words make paired sentences; paired sentences make a palindrome paragraph.

The mirror-pair bank already supplies the algebra: each left half has a
different right half whose letters reverse it.  This module adds the missing
linguistic boundary.  A pair is admitted only when both halves pass the same
whole-sentence grammar gate, and rendering preserves those boundaries instead
of asking a punctuation optimiser to cut across unrelated chunks afterward.
"""
from __future__ import annotations

from collections import Counter
from math import inf
from typing import Optional, Sequence

from .present import SENTENCE, _tier
from .spelling import spell
from .validator import is_palindrome, normalize

CONNECTIVES = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "as", "is", "it",
    "i", "for", "and", "or", "if", "no", "not", "was", "are", "be",
    "by", "my", "we", "me", "its", "this", "that", "with", "from",
})


def is_sentence_pair(left: Sequence[str], right: Sequence[str],
                     table, shapes, trigrams=None) -> bool:
    """Both mirror halves independently have an attested sentence reading."""
    return (_tier(left, table, shapes, trigrams) == SENTENCE
            and _tier(right, table, shapes, trigrams) == SENTENCE)


def sentence_pairs(pairs, table, shapes, trigrams=None):
    """Keep structural pairs whose two textual halves are sentences."""
    return [pair for pair in pairs
            if is_sentence_pair(pair[0], pair[1], table, shapes, trigrams)]


def sentence_centres(rows, table, shapes, trigrams=None):
    """Keep palindromic centre rows that independently read as sentences."""
    return [row for row in rows
            if _tier(row["words"], table, shapes, trigrams) == SENTENCE]


def render_layout(layout: Sequence[dict]) -> tuple[str, list[dict]]:
    """Render exactly one sentence per structural layout slot.

    Punctuation is letter-invisible, but sentence ownership is not.  Keeping a
    sentence inside one chunk prevents the old presenter from merging material
    from unrelated mirror-pairs or cutting the same short run more than once.
    """
    sentences = []
    for chunk in layout:
        words = chunk["text"].split()
        text = spell(words)
        sentences.append({"slot": chunk["slot"], "role": chunk["role"],
                          "text": text, "source": chunk["source"],
                          "words": len(words)})
    rendered = " ".join(sentence["text"] for sentence in sentences)
    plain = " ".join(chunk["text"] for chunk in layout)
    assert normalize(rendered) == normalize(plain), "hierarchy changed letters"
    assert is_palindrome(rendered), "hierarchical rendering broke the mirror"
    return rendered, sentences


def _features(words: Sequence[str]) -> tuple[list[tuple[str, str]], tuple[int, ...]]:
    """Repetition features inside one sentence; seams are punctuation."""
    return list(zip(words, words[1:])), tuple(len(word) for word in words)


def select_sentence_pairs(pairs, centre_words: Sequence[str], target_letters: int,
                          max_pairs: Optional[int] = None,
                          max_template_uses: int = 2,
                          max_bigram_uses: int = 2,
                          max_content_word_uses: int = 3) -> tuple[list[dict], dict]:
    """Select pairs under hard anti-cycle constraints.

    Strange, compressed sentences remain eligible. What is excluded is the
    mechanical failure mode: adjacent duplicate words, a bigram repeated more
    than twice, or one word-length template repeated throughout the paragraph.
    The rules are boolean, so there is no scalar reward to hack.
    """
    used = []
    bigrams: Counter = Counter()
    templates: Counter = Counter()
    content_words: Counter = Counter(
        word for word in centre_words if word not in CONNECTIVES)
    rejected: Counter = Counter()
    letters = sum(len(word) for word in centre_words)
    pair_cap = max_pairs if max_pairs is not None else inf
    centre_bigrams, centre_template = _features(centre_words)
    bigrams.update(centre_bigrams)
    if centre_words:
        templates.update([centre_template])

    for left, right, source in pairs:
        if len(used) >= pair_cap:
            break
        add = 2 * sum(len(word) for word in left)
        if letters + add > target_letters:
            continue
        sentence_features = [_features(left), _features(right)]
        candidate_bigrams = [pair for features, _ in sentence_features for pair in features]
        candidate_templates = [template for _, template in sentence_features]
        candidate_content = [word for sentence in (left, right) for word in sentence
                             if word not in CONNECTIVES]
        if any(a == b for a, b in candidate_bigrams):
            rejected["adjacent_word"] += 1
            continue
        if any(bigrams[pair] + candidate_bigrams.count(pair) > max_bigram_uses
               for pair in set(candidate_bigrams)):
            rejected["bigram"] += 1
            continue
        if any(content_words[word] + candidate_content.count(word)
               > max_content_word_uses for word in set(candidate_content)):
            rejected["content_word"] += 1
            continue
        if any(templates[template] + candidate_templates.count(template)
               > max_template_uses for template in set(candidate_templates)):
            rejected["template"] += 1
            continue
        used.append({"left": list(left), "right": list(right), "source": source})
        bigrams.update(candidate_bigrams)
        templates.update(candidate_templates)
        content_words.update(candidate_content)
        letters += add
        if letters >= target_letters:
            break

    return used, {"rejected": dict(rejected),
                  "repeated_bigrams": sum(count - 1 for count in bigrams.values()),
                  "max_bigram_uses": max(bigrams.values(), default=0),
                  "max_template_uses": max(templates.values(), default=0),
                  "max_content_word_uses": max(content_words.values(), default=0),
                  "unique_bigrams": len(bigrams)}
