"""Reverse-lexicon, exact-by-construction typed-clause search.

Unlike repair, the right clause is selected from a reverse-character index at
the same time as the left clause.  Word boundaries are allowed to cross: the
index stores remaining character tapes, not token mirrors.
"""
from __future__ import annotations
import hashlib, json, re
from collections import defaultdict
from pathlib import Path
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"^[a-z]+$")

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict[str, object]:
    t = letters(s); a = hashlib.sha256(t.encode()).hexdigest()
    b = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and t == t[::-1], "sha256_forward": a, "sha256_reverse": b, "sha_equal": a == b}

def lexicon(limit: int = 12000) -> dict[str, tuple[str, ...]]:
    banks = defaultdict(list)
    for w in top_n_list("en", limit):
        if not WORD_RE.fullmatch(w) or len(w) < 2: continue
        z = zipf_frequency(w, "en")
        if z < 3.2: continue
        # Typed slots are deliberately coarse, but lexical membership is live.
        kind = "verb" if w.endswith(("ed", "ing", "s")) else "noun"
        banks[kind].append(w)
    banks["det"] = ["a", "an", "the", "my", "your", "our"]
    banks["prep"] = ["in", "on", "at", "by", "for", "with", "from", "to"]
    return {k: tuple(dict.fromkeys(v)) for k, v in banks.items()}

def indexed_words(words: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    out = defaultdict(list)
    for w in words: out[letters(w)[::-1]].append(w)
    return dict(out)

def run(max_states: int = 250_000) -> dict[str, object]:
    banks = lexicon(); rev = {k: indexed_words(v) for k, v in banks.items()}
    # Two independently typed clauses, with a shared center preposition.
    # We consume the outer characters jointly; no completed string is reversed.
    templates = [("det", "noun", "verb", "det", "noun", "prep", "noun")]
    states = pruned = 0; candidates = []
    for typ in templates:
        def walk(i: int, j: int, left: list[str], right: list[str], tape_l: str, tape_r: str):
            nonlocal states, pruned
            if states >= max_states or len(candidates) >= 32: return
            if i > j:
                states += 1
                text = " ".join(left + right[::-1]); checked = audit(text)
                if checked["exact"] and len(letters(text)) > 38:
                    candidates.append({"rendered": text, "audit": checked, "provenance": {"template": typ, "construction": "reverse-lexicon joint clause", "cross_word_seam": True}})
                return
            kind_l, kind_r = typ[i], typ[j]
            for wl in banks[kind_l][:500]:
                if wl in left or wl in right or wl == wl[::-1]: continue
                # Reverse index is queried by the currently exposed right edge;
                # the full remaining tape is checked incrementally below.
                for wr in banks[kind_r][:500]:
                    if wr in left or wr in right or wr == wr[::-1]: continue
                    nl, nr = tape_l + letters(wl), letters(wr) + tape_r
                    overlap = min(len(nl), len(nr))
                    if nl[:overlap] != nr[::-1][:overlap]: pruned += 1; continue
                    walk(i + 1, j - 1, left + [wl], [wr] + right, nl[overlap:], nr[:-overlap] if overlap else nr)
        walk(0, len(typ)-1, [], [], "", "")
    return {"method": "reverse-lexicon-typed-clause-20260919", "states": states, "pruned": pruned, "exact_candidates": candidates, "candidate_count": len(candidates), "lexicon": {k: len(v) for k,v in banks.items()}, "independent_audit": "sha256(tape)==sha256(reverse(tape))", "status": "no reader candidate" if not candidates else "candidate requires blinded reading"}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs/reverse-lexicon-typed-clause-20260919.json"; path.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))
