"""v3: one palindrome, made of real words, assembled from chunks.

What v3 serves and why it is shaped this way
--------------------------------------------
One generation. Not a paragraph, not a stream — a single palindrome, written
out the way a person would write it, with the chunks it was assembled from
exposed so the structure is inspectable rather than asserted.

Three decisions, each forced by a measurement rather than a preference.

**Real words.** Every unit is checked against `lexicon.is_real_word`, so `utc` and
`ips` cannot appear, and against `shortwords.is_real_short`, so the one- and
two-letter filler a frequency list offers cannot either. v1's output is 52.6%
one- and two-letter words against real English's 18.5%, and that is most of
what makes it unreadable.

**Chunked, not searched.** A free-running search reaches any length easily —
the deployed v1 closes 958 letters in 14 seconds — and does not read at any
length. Assembly from units that pay the mirror internally is the only
structure that survives, and it makes length a non-problem
(`experiments/RESULTS-extend.md`).

**No automatic growth, by default.** The wrap operations reliably make a
palindrome longer: 750 of 750 seeds grew, mean +36 letters. Two blind
annotators, position balanced, with 6/6 on calibration and complete agreement,
then preferred the SEED on 20 of 20 pairs. Growth is a length hack. It is
available at `?grow=N` because it is a real capability and the numbers are
published, and it is off unless asked for.

The corpus is verified, not generated on demand
-----------------------------------------------
Serving requires a palindrome now, and the good material comes from walks of
millions of candidates that take minutes on 32 cores. So v3 serves from a bank
that was found offline and is verified again on load and on every request. Two
sources, and the response says which:

    generated   found by this project's own enumeration, including the
                Polaris run of 23 August 2026
    catalogue   palindromes the record already contained

`novel=true` restricts to the first, which is the honest default: a paragraph
that reads well because somebody else wrote the sentences is the shortcut
`docs/NORTH-STAR.md` exists to name.

What the name does and does not claim
-------------------------------------
`docs/NORTH-STAR.md` reserved "v3" for the goal — v1's structure with v2's
readability — before any code carried it. This module now carries it, so what
it clears has to be written down beside it rather than inferred from the
number. Measured over 24 seeds at each length (`experiments/RESULTS-north-star-v3.md`):

    letters      c1    c2    c3     c4     c5     c9
    400        24/24 24/24 24/24  21/24  23/24  24/24
    1,200      24/24 24/24 24/24   1/24  22/24  24/24
    4,000      24/24 24/24 24/24   0/24  11/24  24/24
    14,500     24/24 24/24 14/24   0/24   0/24  24/24

Criterion 4 — no sentence repeats — is the one that fails, and it fails for a
reason that is worth stating precisely: **no CHUNK ever repeats, and that is
not the same property.** Punctuation is applied to the assembled word run
rather than per chunk, so two unrelated chunks containing the same short word
run get cut into the same sentence; `bar a met` turns up in four of them at
4,000 letters. The assembly is sound and the presentation collides, which
means the fix belongs in `present.py` or in the chunk selection, not here.

Criteria 6, 7 and 8 — grammatical, has a subject, reads as prose — are NOT
claimed. They need blind judging with salad and real-prose controls. Four
automated proxies have disagreed with blind ranking in this project and none
has ever agreed on it.
"""
from __future__ import annotations

import glob
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from llm_palindrome.lexicon import is_real_word, load_lexicon
from llm_palindrome.present import present
from llm_palindrome.shortwords import is_real_short
from llm_palindrome.validator import is_palindrome, normalize

router = APIRouter(prefix="/api/v3")

BANK_PATH = os.environ.get("PALINDROME_V3_BANK", "data/v3_bank.json")
MIN_LETTERS = int(os.environ.get("PALINDROME_V3_MIN", "16"))
# What production announces: server/app.py plans for 3 * LENGTH_FLOOR letters.
TARGET_LETTERS = int(os.environ.get("PALINDROME_V3_TARGET", "1200"))

_bank: list[dict] = []
_tables = None
_load_error: Optional[str] = None


# ----------------------------------------------------------------- vocabulary

LEXICON_PATH = os.environ.get("PALINDROME_LEXICON", "data/lexicon.txt")
_lexicon: Optional[frozenset] = None


def lexicon() -> frozenset:
    global _lexicon
    if _lexicon is None:
        _lexicon = load_lexicon(LEXICON_PATH)
    return _lexicon


def real_words(words) -> bool:
    """Every unit a word a reader accepts.

    Two filters, and both earn their place. `is_real_word` rejects strings the
    frequency list contains and English does not — "utc", "ips". `is_real_short`
    rejects the one- and two-letter strings that fit any overhang and are how a
    search cheats when the letters get awkward; v1 output is 52.6% of those
    against real English's 18.5%.
    """
    lex = lexicon()
    return all(w.isalpha() and is_real_word(w, lex) and is_real_short(w)
               for w in words)


# ----------------------------------------------------------------------- bank

def _load_bank(path: str = BANK_PATH) -> list[dict]:
    """Read the bank and re-verify every entry.

    A stored palindrome that is not one is the single worst thing this service
    could serve, so the check is repeated on load rather than trusted from the
    file that claims it.
    """
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        text = r["text"]
        words = text.split()
        if not is_palindrome(text):
            continue
        if len(normalize(text)) < MIN_LETTERS:
            continue
        if not real_words(words):
            continue
        out.append({"text": text, "words": words,
                    "letters": len(normalize(text)),
                    "source": r.get("source", "generated"),
                    "origin": r.get("origin", "")})
    return out


def ensure_loaded() -> None:
    global _bank, _tables, _load_error
    if _bank or _load_error:
        return
    try:
        _bank = _load_bank()
        from llm_palindrome.syntax import brown_tables
        _tables = brown_tables()
        if not _bank:
            _load_error = "bank empty after verification"
    except Exception as exc:                     # noqa: BLE001 - reported, not raised
        _load_error = f"{type(exc).__name__}: {exc}"


# -------------------------------------------------------------------- chunks

def chunks_of(words) -> list[dict]:
    """Split into the mirrored halves and the centre, which is the structure.

    A palindrome's letters split at the midpoint into `L` and `reverse(L)`. The
    word boundary nearest that midpoint is where a reader would say the text
    turns, and a unit straddling it is the centre. Exposing this is what makes
    "assembled from chunks" checkable instead of a claim.
    """
    letters = [len(w) for w in words]
    half = sum(letters) / 2
    run = 0
    left, centre, right = [], None, []
    for w, n in zip(words, letters):
        if run + n <= half:
            left.append(w)
        elif run >= half:
            right.append(w)
        else:
            centre = w
        run += n
    out = [{"role": "left", "text": " ".join(left)}]
    if centre:
        out.append({"role": "centre", "text": centre})
    out.append({"role": "right", "text": " ".join(right)})
    return [c for c in out if c["text"]]


def grow(words, n: int):
    """Apply `n` mirror-preserving wraps. Off by default; see the module note."""
    from experiments.extend_ops import glue_pairs, material, grow as _grow
    from llm_palindrome.generate import build_vocab
    selfpal, pairs = material(build_vocab(30000))
    table, shapes, _ = _tables
    out, trail = _grow(list(words), selfpal, pairs, table, shapes,
                       max_ops=n, glue=glue_pairs())
    return out, trail


# --------------------------------------------------------------- composition

def harvest_pair(words) -> Optional[tuple[list[str], list[str]]]:
    """Split a palindrome into its two halves, which mirror each other.

    A palindrome's letters are `L + reverse(L)`, so if a word boundary falls at
    the exact midpoint the two halves are a mirror-pair made of DIFFERENT text.
    501 of the 540 bank entries split this way; the rest have the turn inside a
    word and can only ever serve as a centre.
    """
    lengths = [len(w) for w in words]
    half = sum(lengths) / 2
    run = 0
    for i, n in enumerate(lengths):
        if run == half:
            left, right = list(words[:i]), list(words[i:])
            if left and right and normalize(" ".join(left)) == \
                    normalize(" ".join(right))[::-1]:
                return left, right
            return None
        run += n
    return None


def capacity(pairs, centre_letters: int = 0) -> int:
    """Longest palindrome this material can build: every pair, both halves."""
    return centre_letters + 2 * sum(
        len(normalize(" ".join(left))) for left, _, _ in pairs)


def compose(pairs, centre, target_letters: int,
            max_chops: Optional[int] = None) -> list[dict]:
    """Nest mirror-pairs around a centre: L1 L2 ... C ... R2 R1.

    Why this arrangement and not simply concatenating whole palindromes: a
    string of self-palindromic units is a palindrome only when the SEQUENCE of
    units is itself a palindrome, so unit k must equal unit n+1-k and every
    unit but the centre appears twice. That is forced by the algebra rather
    than chosen, and it is why the earlier paragraph endpoint repeats itself.

    Mirror-pairs avoid it. What returns at position n+1-k is the OTHER half,
    which is different text, so nothing repeats and the whole still mirrors.
    """
    used: list[dict] = []
    letters = len(normalize(" ".join(centre["words"])))
    for left, right, src in pairs:
        if max_chops is not None and len(used) >= max_chops:
            break
        add = 2 * len(normalize(" ".join(left)))
        if letters + add > target_letters:
            continue
        used.append({"left": left, "right": right, "source": src})
        letters += add
        if letters >= target_letters:
            break
    return used


def assemble(used, centre) -> tuple[list[str], list[dict]]:
    """Lay the chosen pairs out around the centre and describe the layout."""
    words: list[str] = []
    layout: list[dict] = []
    for i, u in enumerate(used):
        words += u["left"]
        layout.append({"slot": i, "role": "left", "text": " ".join(u["left"]),
                       "source": u["source"]})
    words += centre["words"]
    layout.append({"slot": len(used), "role": "centre",
                   "text": " ".join(centre["words"]),
                   "source": centre["source"]})
    for i, u in enumerate(reversed(used)):
        words += u["right"]
        layout.append({"slot": len(used) + 1 + i, "role": "right",
                       "text": " ".join(u["right"]), "source": u["source"]})
    return words, layout


@router.get("/composition")
def composition(seed: Optional[int] = Query(None),
                letters: int = Query(TARGET_LETTERS, ge=40, le=20000,
                                     description="target length; the bank "
                                                 "caps it, see /health"),
                chops: Optional[int] = Query(None, ge=1, le=600,
                                             description="cap the number of "
                                                         "mirror-pairs used"),
                longest_first: bool = Query(False,
                                            description="prefer long pairs, "
                                                        "so fewer seams"),
                centre: Optional[str] = Query(None, max_length=200,
                                              description="your own palindrome, "
                                                          "used as the centre"),
                novel: bool = Query(True)):
    """Many mirror-pairs nested around one centre.

    Length is free here and that is the point. `/palindrome` serves one short
    verified palindrome; this nests as many as asked for, and the only limits
    are the size of the bank and what is asked of it. The bank currently holds
    about 14,500 letters of material.

    Two dials, because at a fixed length they trade against each other. A
    target of 1,200 letters can be forty short pairs or thirty long ones, and
    every pair boundary is a seam where two unrelated fragments meet.
    `longest_first` spends the bank on fewer, longer chunks; `chops` caps the
    count outright. Fewer seams is the only quality lever this endpoint has,
    and it has not been judged, so neither dial is claimed to read better.

    `centre` takes the visitor's own palindrome and puts it in the one slot the
    algebra lets an arbitrary palindrome occupy. Every other position is half
    of a pair and is fixed by its opposite number; the centre is fixed by
    nothing, so it is the only place a text that was not walked out of the bank
    can go without breaking the mirror. It is rejected rather than repaired if
    its letters do not read both ways.
    """
    ensure_loaded()
    if _load_error:
        raise HTTPException(status_code=503, detail=_load_error)

    pool = [r for r in _bank if not novel or r["source"] == "generated"]
    # Deduplicate on the half text. Two different bank entries can share a
    # half — the 27-31 letter band is dominated by a few centres, so many
    # entries are rewordings around the same core — and admitting both puts
    # the same chunk in twice. At the full 14,500 letters that produced one
    # repeat, which is one more than this structure is allowed.
    pairs = []
    seen_halves: set[str] = set()
    for row in pool:
        got = harvest_pair(row["words"])
        if not got:
            continue
        left_k = normalize(" ".join(got[0]))
        right_k = normalize(" ".join(got[1]))
        # A half that is itself a palindrome is a degenerate pair: its mirror
        # is itself, so the same text lands at both mirrored positions and the
        # scheme's whole advantage over self-palindromic units is lost.
        # `no is ice decision` is one, and it produced the only repeat at full
        # length.
        if left_k == right_k:
            continue
        if left_k in seen_halves or right_k in seen_halves:
            continue
        seen_halves.add(left_k)
        seen_halves.add(right_k)
        pairs.append((got[0], got[1], row["source"]))
    # Any bank entry can be the centre: it is a palindrome, which is the only
    # requirement the algebra makes of that slot. An earlier version reserved
    # the entries that do NOT split for it, which left no centre at all once
    # `novel` filtered the pool down to entries that all split.
    centres = pool
    if not pairs or not centres:
        raise HTTPException(status_code=404, detail="not enough material")

    rng = random.Random(seed if seed is not None else time.time_ns())
    rng.shuffle(pairs)
    if longest_first:
        pairs.sort(key=lambda t: -len(normalize(" ".join(t[0]))))
    if centre is not None and centre.strip():
        # The visitor's own. Checked, never fixed up: silently trimming a
        # near-palindrome into a real one would hand back something they did
        # not write and call it theirs.
        own = centre.strip().split()
        if not normalize(" ".join(own)):
            raise HTTPException(status_code=400,
                                detail="that has no letters in it")
        if not is_palindrome(" ".join(own)):
            raise HTTPException(
                status_code=400,
                detail="that is not a palindrome — its letters have to read "
                       "the same both ways")
        centre_dict = {"words": own, "source": "yours"}
    else:
        # The centre must not duplicate a half either.
        centre_row = rng.choice(centres)
        for _ in range(20):
            if normalize(" ".join(centre_row["words"])) not in seen_halves:
                break
            centre_row = rng.choice(centres)
        centre_dict = {"words": centre_row["words"],
                       "source": centre_row["source"]}
    centre = centre_dict

    cap = capacity(pairs, len(normalize(" ".join(centre["words"]))))
    used = compose(pairs, centre, min(letters, cap), chops)
    words, layout = assemble(used, centre)
    text = " ".join(words)
    if not is_palindrome(text):
        raise HTTPException(status_code=500, detail="assembly broke the mirror")

    table, shapes, trig = _tables
    written = present(words, table, shapes, trig)
    if normalize(written) != normalize(text):
        raise HTTPException(status_code=500,
                            detail="presentation changed the letters")

    texts = [c["text"] for c in layout]
    return {
        "version": 3,
        "text": written,
        "plain": text,
        "letters": len(normalize(text)),
        "words": len(words),
        "pairs": len(used),
        "requested_letters": letters,
        "capacity_letters": cap,
        "chunks": layout,
        "distinct_chunks": len(set(texts)),
        "repeats": len(texts) - len(set(texts)),
        "centre_is_yours": centre["source"] == "yours",
        "notes": {
            "structure": "L1 L2 ... C ... R2 R1, where Ri is Li's letters "
                         "reversed and is therefore different text",
            "why": "a sequence of self-palindromic units forces unit k to "
                   "equal unit n+1-k; mirror-pairs do not repeat",
        },
    }


# ------------------------------------------------------------------ endpoint

@router.get("/palindrome")
def palindrome(seed: Optional[int] = Query(None, description="fix the choice"),
               min_letters: int = Query(MIN_LETTERS, ge=8, le=200),
               novel: bool = Query(True, description="exclude catalogued ones"),
               grow_ops: int = Query(0, ge=0, le=6, alias="grow")):
    """One palindrome, presented, with its chunks and its provenance."""
    ensure_loaded()
    if _load_error:
        raise HTTPException(status_code=503, detail=_load_error)

    pool = [r for r in _bank if r["letters"] >= min_letters]
    if novel:
        pool = [r for r in pool if r["source"] == "generated"]
    if not pool:
        raise HTTPException(status_code=404,
                            detail="no palindrome matches those constraints")

    rng = random.Random(seed if seed is not None else time.time_ns())
    row = rng.choice(pool)
    words = list(row["words"])
    trail: list[str] = []

    if grow_ops:
        words, trail = grow(words, grow_ops)

    text = " ".join(words)
    if not is_palindrome(text):
        raise HTTPException(status_code=500, detail="assembly broke the mirror")

    table, shapes, trig = _tables
    written = present(words, table, shapes, trig)
    # Presentation may not touch the letters. Checked here as well as inside
    # `present`, because this is the value that leaves the process.
    if normalize(written) != normalize(text):
        raise HTTPException(status_code=500, detail="presentation changed the letters")

    return {
        "version": 3,
        "text": written,
        "plain": text,
        "letters": len(normalize(text)),
        "words": len(words),
        "chunks": chunks_of(words),
        "source": row["source"],
        "origin": row["origin"],
        "grown": bool(trail),
        "operations": trail,
        "notes": {
            "mirror": "letters read identically both ways; case, spaces and "
                      "punctuation are invisible to the check",
            "growth": "wrap operations lengthen reliably and read worse: two "
                      "blind annotators preferred the ungrown seed 20 of 20",
        },
    }


@router.get("/health")
def health():
    """Bank size and, for a slider, the longest composition it can build."""
    ensure_loaded()
    caps = {}
    if not _load_error:
        for novel in (True, False):
            pool = [r for r in _bank if not novel or r["source"] == "generated"]
            pairs = [(g[0], g[1], r["source"]) for r in pool
                     if (g := harvest_pair(r["words"]))]
            caps["novel" if novel else "all"] = {
                "pairs": len(pairs), "max_letters": capacity(pairs)}
    return {"ok": _load_error is None, "version": 3,
            "bank": len(_bank),
            "generated": sum(1 for r in _bank if r["source"] == "generated"),
            "catalogue": sum(1 for r in _bank if r["source"] == "catalogue"),
            "capacity": caps,
            "error": _load_error}
