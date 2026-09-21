"""Live character-seam search over complete, authored English frame pairs.

Unlike a finished-tape reverse lookup, this enumerates ordinary forward
subject/verb/object frames on both sides while consuming the outer character
orbit.  It is intentionally narrow enough that every emitted row can be read
as two clauses, and broad enough to test the seam pattern behind the accepted
38-letter anchor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import brown
from wordfreq import zipf_frequency


TAG = {
    "nn": "NOUN", "nns": "NOUN", "np": "NAME", "nps": "NAME",
    "nr": "NAME", "vb": "VERB", "vbd": "VERB", "vbg": "VERB",
    "vbn": "VERB", "vbz": "VERB", "md": "VERB", "jj": "ADJ",
    "jjs": "ADJ", "jjr": "ADJ", "rb": "ADV", "rbr": "ADV",
    "rbt": "ADV", "dt": "DET", "at": "DET", "dti": "DET",
    "dts": "DET", "pp": "PRON", "pps": "PRON", "ppo": "PRON",
    "ppss": "PRON", "ppl": "PRON", "in": "PREP", "to": "PREP",
    "cc": "CONJ", "cs": "COMP",
}

MANUAL_NAMES = (
    "diana nora aron leon liam mara aram damon nomad pam elba iris anna "
    "evita eliot otto ada eva eve emil lisa naomi neil noel "
    "marge norah sharon dennis edna nell lena alice carol jane reed dave "
    "nina jo ira sara mario jan ina lily arne bette dan reba diane lynn "
    "ed del rena joel lara cecil aaron flora tina arden ellen natasha "
).split()

DETS = ("a an the some no one two three four five six seven eight nine ten"
        " many each every our my his her their this that").split()
NUMS = ("one two three four five six seven eight nine ten"
        " once twice many some").split()

AUTHORED_NOUNS = (
    "aide aides memo memos man men woman women child children clerk "
    "teacher scholar poet sailor pilot baker nurse doctor artist writer "
    "letter letters note notes map maps book books card cards tale tales "
    "story stories lesson lessons answer answers signal signals parcel "
    "parcels garden harbor river bridge road house room window door "
    "stone star rain boat boats bird birds dog cat rat rats wolf deer "
    "apple apples orange oranges flower flowers song songs sonnet sonnets "
    "name names number numbers message messages telegram telegrams diary "
    "diaries paper papers page pages poem poems question questions idea ideas "
    "bard bards tang gnat gnats dirt pot pots toilet toilets "
).split()
AUTHORED_ADJECTIVES = (
    "putrid sad drab top deep far old red new quiet bright "
).split()
AUTHORED_VERBS = (
    "am is are was were be been do does did can will would notes note rips inspire inspires write writes read reads send sends mail mails let lets "
    "draw draws map maps make makes mark marks hold holds keep keeps see "
    "sees saw say says tell tells teach teaches guide guides carry carries "
    "open opens close closes find finds mend mends plant plants water waters "
    "watch watches hear hears call calls name names love loves save saves "
    "move moves visit visits guard guards record records answer answers "
    "deliver delivers gather gathers repair repairs study studies remember "
    "remembers notice notices carry carries bring brings build builds "
    "emanate emanates emanating assign assigns upset "
).split()

TEMPLATES = [
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "ADJ", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "NUM", "NOUN"),
    ("NAME", "VERB", "NAME", "NOUN"),
    ("NAME", "VERB", "NAME", "VERB", "NAME", "NOUN"),
    ("NAME", "VERB", "NAME", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "VERB", "NAME"),
    ("NAME", "VERB", "NAME", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "NAME"),
    ("VERB", "NAME", "NOUN"),
    ("NAME", "VERB", "NAME", "VERB"),
    ("VERB", "NAME", "VERB", "NAME", "NOUN"),
    ("NAME", "VERB", "NAME", "VERB", "NAME"),
    ("VERB", "NAME", "VERB", "NAME", "NOUN", "PREP"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"),
]

COORDINATED_TEMPLATES = [
    ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ",
     "DET", "NOUN", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "CONJ",
     "DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "CONJ",
     "NAME", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN", "CONJ",
     "DET", "NOUN", "VERB", "NAME"),
]

FULL_TEMPLATES = [
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "ADJ", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NAME"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "NUM", "NOUN"),
    ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN"),
    ("NAME", "VERB", "DET", "NOUN", "COMP", "DET", "NOUN"),
]

# A separate long-form lane for ordinary literary sentence shapes.  The
# initial and pronoun slots deliberately admit one-letter words: excluding
# them makes the search unable to reach natural forms such as "I" and
# initials, and silently biases it toward the shorter anchor's vocabulary.
POETIC_TEMPLATES = [
    ("INITIAL", "NAME", "NOUN", "NOUN", "VERB", "ADJ", "NOUN", "VERB"),
    ("VERB", "ADJ", "PRON", "VERB", "PRON", "DET", "NOUN", "NOUN",
     "NOUN", "VERB", "PREP", "ADJ", "NOUN", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN", "PREP",
     "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "PRON", "VERB", "DET", "NOUN", "PREP",
     "DET", "NOUN"),
]


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1])
                     for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def hidden_span(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.casefold())
    full = norm(text)
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = "".join(words[i:j])
            if 1 < len(span) < len(full) and span == span[::-1]:
                return True
    return False


def lexical_bank(limit: int | None = None) -> tuple[dict[str, tuple[str, ...]], Counter]:
    if limit is None:
        limit = int(os.environ.get("SEAM_LEX_LIMIT", "180"))
    counts: dict[str, Counter] = defaultdict(Counter)
    bigrams: Counter = Counter()
    for word, tag in brown.tagged_words():
        word = word.casefold()
        if not re.fullmatch(r"[a-z]+", word):
            continue
        category = TAG.get(tag.split("-", 1)[0].casefold())
        if category:
            counts[category][word] += 1
    for sentence in brown.sents():
        words = [re.sub(r"[^a-z]", "", word.casefold()) for word in sentence]
        words = [word for word in words if word]
        bigrams.update(zip(words, words[1:]))
    bank = {}
    for category, counter in counts.items():
        bank[category] = tuple(
            word for word, _ in counter.most_common(limit)
            if (len(word) > 1 or category in {"PRON", "INITIAL"})
            and zipf_frequency(word, "en") >= 2.4
        )
    bank["NOUN"] = tuple(dict.fromkeys(AUTHORED_NOUNS + list(bank.get("NOUN", ()))))
    bank["ADJ"] = tuple(dict.fromkeys(AUTHORED_ADJECTIVES + list(bank.get("ADJ", ()))))
    bank["VERB"] = tuple(dict.fromkeys(AUTHORED_VERBS + list(bank.get("VERB", ()))))
    bank["DET"] = tuple(dict.fromkeys(DETS + list(bank.get("DET", ()))))
    bank["NUM"] = tuple(dict.fromkeys(NUMS + list(bank.get("NUM", ()))))
    bank["NAME"] = tuple(word for word in dict.fromkeys(
        MANUAL_NAMES + list(bank.get("NAME", ())))
        if word != "de")
    bank["PRON"] = tuple(dict.fromkeys(
        "i id it we he she they you me us them my our your his her their".split()
        + list(bank.get("PRON", ()))
    ))
    bank["INITIAL"] = tuple("a i j k l m n o p q r s t u v w x y".split())
    return bank, bigrams


@dataclass(frozen=True)
class State:
    li: int
    ri: int
    lw: str
    rw: str
    lp: int
    rp: int
    left: tuple[str, ...]
    right: tuple[str, ...]
    score: float


def search_pair(left_shape: tuple[str, ...], right_shape: tuple[str, ...],
                bank: dict[str, tuple[str, ...]], bigrams: Counter,
                beam: int = 6000) -> list[dict]:
    # The right clause is kept in final forward order but is emitted from its
    # last word backward, so every transition compares the live outer orbit.
    start = State(0, len(right_shape) - 1, "", "", 0, -1, (), (), 0.0)
    frontier = [start]
    closed: list[dict] = []
    seen: set[tuple] = set()

    def score_join(a: str | None, b: str) -> float:
        if b is None:
            return 0.0
        if a is None:
            return math.log1p(zipf_frequency(b, "en"))
        return math.log1p(bigrams[a, b]) + 0.04 * zipf_frequency(b, "en")

    while frontier:
        next_frontier = []
        for state in frontier:
            key = (state.li, state.ri, state.lw, state.rw, state.lp,
                   state.rp, state.left, state.right)
            if key in seen:
                continue
            seen.add(key)
            if (state.li == len(left_shape) and state.ri < 0
                    and not state.lw and not state.rw):
                text = " ".join(state.left) + "; " + " ".join(state.right)
                row = {"rendered": text, "left_shape": list(left_shape),
                       "right_shape": list(right_shape), "audit": audit(text),
                       "hidden_palindromic_span": hidden_span(text),
                       "provenance": "forward-authored frame pair with live character seam"}
                if row["audit"]["exact"]:
                    closed.append(row)
                continue
            left_words = ([state.lw] if state.lw else
                          list(bank.get(left_shape[state.li], ()))
                          if state.li < len(left_shape) else [])
            right_words = ([state.rw] if state.rw else
                           list(bank.get(right_shape[state.ri], ()))
                           if state.ri >= 0 else [])
            for lw in left_words:
                lpos = state.lp if state.lw else 0
                if lpos >= len(lw):
                    continue
                for rw in right_words:
                    rpos = state.rp if state.rw else len(rw) - 1
                    if rpos < 0 or lw[lpos] != rw[rpos]:
                        continue
                    if (not state.lw and lw in state.left + state.right):
                        continue
                    if (not state.rw and rw in state.left + state.right):
                        continue
                    left_done = lpos + 1 == len(lw)
                    right_done = rpos == 0
                    nlw = "" if left_done else lw
                    nrw = "" if right_done else rw
                    nli = state.li + (1 if left_done else 0)
                    nri = state.ri - (1 if right_done else 0)
                    nleft = state.left + ((lw,) if not state.lw else ())
                    nright = ((rw,) if not state.rw else ()) + state.right
                    delta = 0.0
                    if not state.lw:
                        delta += score_join(state.left[-1] if state.left else None, lw)
                    if not state.rw:
                        delta += score_join(rw, state.right[0] if state.right else None)
                    next_frontier.append(State(
                        nli, nri, nlw, nrw,
                        lpos + 1 if not left_done else 0,
                        rpos - 1 if not right_done else -1,
                        nleft, nright, state.score + delta))
        next_frontier.sort(key=lambda s: (-s.score, -len(s.left) - len(s.right)))
        frontier = next_frontier[:beam]
    return closed


def run() -> dict:
    bank, bigrams = lexical_bank()
    rows: list[dict] = []
    beam = int(os.environ.get("SEAM_BEAM", "6000"))
    if os.environ.get("SEAM_POETIC"):
        shapes = POETIC_TEMPLATES
    elif os.environ.get("SEAM_FULL"):
        shapes = FULL_TEMPLATES
    else:
        shapes = COORDINATED_TEMPLATES if os.environ.get("SEAM_COORDINATED") else TEMPLATES
    for left in shapes:
        for right in shapes:
            rows.extend(search_pair(tuple(left), tuple(right), bank, bigrams,
                                    beam=beam))
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    clean = [row for row in rows if not row["hidden_palindromic_span"]]
    return {"experiment": "semantic-seam-frame-search-20260920",
            "method": "forward complete clause-pair product with live outer-character seam",
            "stats": {"templates": len(shapes), "beam": beam,
                      "coordinated": bool(os.environ.get("SEAM_COORDINATED")),
                      "exact": len(rows),
                      "longest_exact": max((r["audit"]["letters"] for r in rows), default=0),
                      "longest_clean_exact": max((r["audit"]["letters"] for r in clean), default=0)},
            "exact_candidates": rows[:200], "clean_exact_candidates": clean[:100],
            "provenance": {"source": "Brown POS counts plus held-out authored names and number words",
                           "finished_tape_reversal": False, "post_hoc_repair": False,
                           "audits": ["two-pointer mismatch", "forward/reverse SHA-256"]}}


if __name__ == "__main__":
    out = Path("runs/semantic-seam-frame-search-20260920.json")
    result = run()
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
    for row in result["clean_exact_candidates"][:20]:
        print(row["audit"]["letters"], row["rendered"])
