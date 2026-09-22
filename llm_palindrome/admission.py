"""Fail-closed mechanical admission for prospective readable palindromes.

This module deliberately establishes only *mechanical* eligibility.  Passing
it never means that a string is readable; that claim needs blinded human
evidence.  Every constructor and serving path must call this gate before it
can expose a prospective item.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
# Paragraph candidates may contain line breaks. ``\s`` admits ordinary layout
# whitespace while the surrounding class still rejects unsupported symbols.
ALLOWED_RENDERING = re.compile(r"[A-Za-z][A-Za-z\s'\-.,;:!?]*")
FORBIDDEN_CATALOGUE_TAPES = frozenset({"margeletsnorahseesharonstelegram"})
# A few classic palindromes are routinely reused as an *outer scaffold*: the
# author retains their distinctive opening and final word(s), then claims a
# newly generated interior.  They are prohibited even when the copied pieces
# are separated by fresh material.  The patterns below are deliberately word
# boundary aware and narrow; ordinary lexical overlap alone is not enough.
FORBIDDEN_CATALOGUE_ENDPOINT_SCAFFOLDS = (
    (("no", "it", "is"), ("position",)),  # No, it is opposition.
    (("see",), ("bees",)),                  # Eva, can I see bees in a cave?
    (("go",), ("dog",)),                    # Go dog.
)
# A dictionary includes abbreviations, solfege labels, surnames, and isolated
# syllables.  These two-letter forms are ordinary English function/content
# words that can plausibly appear in an intact sentence.  This is only a
# fail-closed fragment filter, never a readability certificate.
ORDINARY_TWO_LETTER_WORDS = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or ox so to up us we".split()
)
# Repeating an article, complementizer, pronoun, or preposition is ordinary
# English syntax; treating it as a repeated construction unit makes long,
# readable prose impossible and does not block the actual shortcuts.  Content
# words must still be unique, while repeated multiword sequences remain a
# separate hard rejection below.
REPEATABLE_FUNCTION_WORDS = frozenset(
    "a an the this that these those i me we us you he him she her it they them "
    "who which whose and or but if as while when after before because though "
    "of to in on at by for from with without near during is are was were be been "
    "do does did can could will would may might should have has had not no "
    "some any each every either neither another all both few many much more most "
    "several enough less little own same such whatever whichever".split()
)


def normalize_letters(text: str) -> str:
    """Return the ASCII letter tape, rejecting unsupported alphabetic input."""
    if any(character.isalpha() and not character.isascii() for character in text):
        raise ValueError("unsupported non-ASCII alphabetic character")
    return "".join(re.findall(r"[a-z]", text.casefold()))


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize the permitted English rendering independently of punctuation."""
    return tuple(WORD.findall(text.casefold()))


def has_only_ordinary_short_words(units: tuple[str, ...]) -> bool:
    """Reject common dictionary fragments such as ``la`` or ``st``.

    A passing value says only that a candidate cleared this narrow lexical
    screen.  Whole-sentence readability remains a blinded-reader question.
    """
    return all(
        len(normalize_letters(unit)) != 2
        or normalize_letters(unit) in ORDINARY_TWO_LETTER_WORDS
        for unit in units
    )


def is_boundary_aligned_word_mirror(units: tuple[str, ...]) -> bool:
    """Detect whole-word reverse pairing such as ``rats ... star``.

    Exact character symmetry is required, but this construction is explicitly
    outside the project's target: it reads as a word-order reflection rather
    than independently composed prose.
    """
    normalized = tuple(normalize_letters(unit) for unit in units)
    return bool(normalized) and normalized == tuple(
        unit[::-1] for unit in reversed(normalized)
    )


def has_repeated_nontrivial_unit(units: tuple[str, ...]) -> bool:
    """Reject a repeated contiguous two-or-more-word unit.

    This catches the less obvious ``A B ... A B`` construction without
    banning ordinary repeated syntax such as ``that a`` in nested relative
    clauses. A repeated unit must contain a content word to count as a
    construction shortcut.
    """
    normalized = tuple(normalize_letters(unit) for unit in units)
    for width in range(2, len(normalized) // 2 + 1):
        for left in range(0, len(normalized) - width):
            for right in range(left + width, len(normalized) - width + 1):
                repeated = normalized[left:left + width]
                if (
                    repeated == normalized[right:right + width]
                    and any(word not in REPEATABLE_FUNCTION_WORDS for word in repeated)
                ):
                    return True
    return False


def has_self_palindromic_proper_multiword_span(units: tuple[str, ...]) -> bool:
    """Reject a hidden two-or-more-word palindrome used as a center unit.

    The complete rendered item is necessarily a palindrome, so its full span
    is deliberately excluded.  What this blocks is a *proper* contiguous
    multiword island such as ``to order red root`` embedded in a longer
    output.  Such an island is a preassembled shortcut, even if the words
    outside it were generated character by character.
    """
    normalized = tuple(normalize_letters(unit) for unit in units)
    total = len(normalized)
    for width in range(2, total + 1):
        for start in range(0, total - width + 1):
            end = start + width
            if start == 0 and end == total:
                continue
            tape = "".join(normalized[start:end])
            if tape and tape == tape[::-1]:
                return True
    return False


def has_distinct_content_words(units: tuple[str, ...]) -> bool:
    """Permit ordinary repeated function words, never repeated content words.

    This is a construction anti-shortcut screen, not a readability score.  It
    works together with the multiword-unit check so articles and
    complementizers can recur as grammar requires without permitting repeated
    sentence blocks or copied content.
    """
    content = tuple(
        normalized for normalized in (normalize_letters(unit) for unit in units)
        if normalized and normalized not in REPEATABLE_FUNCTION_WORDS
    )
    return len(content) == len(set(content))


def is_catalogue_family_derivative(units: tuple[str, ...]) -> bool:
    """Reject the slot-relexicalized public ``lets ... see ...'s`` template.

    Scope is intentionally conservative and explicit: any contiguous surface
    shaped ``NAME lets NAME see NAME's NOUN`` is treated as a
    relexicalization of the well-known ``Marge lets Norah see Sharon's
    telegram`` construction.  Checking a window, rather than only an entire
    six-token text, also blocks palindrome-preserving wrappers such as
    ``Part ... telegram trap``.
    """
    for start in range(len(units) - 5):
        window = units[start:start + 6]
        if (
            window[1] == "lets"
            and window[3] == "see"
            # The character tape cannot distinguish ``Sarah's`` from
            # ``Sarahs``. Treat both renderings as the same provenance
            # signature; otherwise apostrophe deletion revives the shortcut.
            and normalize_letters(window[4]).endswith("s")
            and all(normalize_letters(unit) for unit in window)
        ):
            return True
    return False


def has_forbidden_catalogue_endpoint_scaffold(units: tuple[str, ...]) -> bool:
    """Reject copied classic endpoints separated by a newly authored middle.

    Exact candidates may not inherit their outer construction from a known
    palindrome and present only the interior as generated.  This deliberately
    applies before exactness: a non-palindromic control can expose an
    inadmissible construction family without ever becoming a candidate.
    """
    normalized = tuple(normalize_letters(unit) for unit in units)
    return any(
        len(normalized) >= len(prefix) + len(suffix)
        and normalized[:len(prefix)] == prefix
        and normalized[-len(suffix):] == suffix
        for prefix, suffix in FORBIDDEN_CATALOGUE_ENDPOINT_SCAFFOLDS
    )


@lru_cache(maxsize=1)
def _default_lexicon() -> frozenset[str]:
    """Load the fixed dictionary once for a fail-closed lexical gate."""
    from .lexicon import load_lexicon

    return load_lexicon(str(Path(__file__).resolve().parents[1] / "data" / "lexicon.txt"))


@lru_cache(maxsize=1)
def _default_catalogue() -> frozenset[str]:
    """Load the repository's fixed exclusion catalogue.

    This is part of central admission, rather than an optional caller policy:
    a candidate author must never be able to omit the known material it needs
    to exclude. Callers may add project-local exclusions below, never replace
    this baseline.
    """
    import json

    path = Path(__file__).resolve().parents[1] / "data" / "known_palindromes.json"
    return frozenset(json.loads(path.read_text()))


def is_lexical_word(unit: str) -> bool:
    """Require a conventional dictionary-backed English lexical form.

    Admission deliberately has no caller-supplied proper-name exception:
    otherwise a generator can label arbitrary character noise a name and
    bypass the shared lexical gate.  A future named-entity policy must be a
    versioned, independently auditable registry rather than candidate-author
    input.
    """
    from .lexicon import is_real_word

    normalized = normalize_letters(unit)
    return is_real_word(normalized, _default_lexicon())


def mechanical_admission_checks(
    text: str,
    *,
    local_catalogue: Iterable[str] = (),
    min_letters: int = 30,
    max_letters: int = 100,
) -> dict[str, bool]:
    """Return every fail-closed mechanical acceptance check.

    The result intentionally does not contain a readability score.  A caller
    may admit an item only when every value is true, then must still route it
    to the blinded human study gate.
    """
    try:
        tape = normalize_letters(text)
        units = tokenize(text)
        supported_ascii_letters = True
    except ValueError:
        tape, units, supported_ascii_letters = "", (), False
    # Callers may supply rendered catalogue strings or already-normalized
    # tapes. Normalize both forms so spaces and punctuation cannot open a
    # catalogue-absence bypass.
    known = set()
    for item in (*_default_catalogue(), *local_catalogue):
        try:
            known.add(normalize_letters(item))
        except (TypeError, ValueError):
            # Malformed catalogue input cannot make a candidate pass. Candidate
            # normalization itself remains fail-closed above.
            continue
    normalized_units = tuple(normalize_letters(unit) for unit in units)
    return {
        "supported_ascii_letters": supported_ascii_letters,
        "nonempty": bool(tape and units),
        "exact_letter_palindrome": bool(tape) and tape == tape[::-1],
        "length_band": min_letters <= len(tape) <= max_letters,
        "word_form": bool(ALLOWED_RENDERING.fullmatch(text)),
        "lexicon_words": bool(units) and all(is_lexical_word(unit) for unit in units),
        "ordinary_short_words": bool(units) and has_only_ordinary_short_words(units),
        # Compatibility name retained for constructor artifacts.  It now
        # means content-word distinctness; repeated function words are normal
        # prose, while repeated multiword units are still hard-rejected.
        "distinct_words": has_distinct_content_words(units),
        # One-letter articles/pronouns and ordinary function words can be
        # character-palindromic without forming a self-palindromic
        # construction unit. Content-word palindromes remain disallowed.
        "no_self_palindromic_word": all(
            unit != unit[::-1] or unit in REPEATABLE_FUNCTION_WORDS
            for unit in normalized_units
        ),
        "not_word_order_symmetry": not is_boundary_aligned_word_mirror(units),
        "no_repeated_nontrivial_unit": not has_repeated_nontrivial_unit(units),
        "no_self_palindromic_proper_multiword_span": not has_self_palindromic_proper_multiword_span(units),
        "not_forbidden_catalogue_control": tape not in FORBIDDEN_CATALOGUE_TAPES,
        "not_catalogue_family_derivative": not is_catalogue_family_derivative(units),
        "not_forbidden_catalogue_endpoint_scaffold": not has_forbidden_catalogue_endpoint_scaffold(units),
        "absent_from_local_catalogue": tape not in known,
        # Kept as a compatibility alias for proposal records written before the
        # shared gate; both names describe the same hard exclusion.
        "local_catalogue_absent": tape not in known,
    }
