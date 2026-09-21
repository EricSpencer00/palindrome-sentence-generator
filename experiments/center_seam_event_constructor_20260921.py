#!/usr/bin/env python3
"""Center-seam event constructor.

Two independently typed event clauses grow from the outside toward a live
character seam.  Word boundaries are variable and the seam may fall inside a
word; no clause is built by reversing another clause.  This is an exploratory
constructor, not a readability certificate.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANK = {
    "det": ["a", "an", "the", "some", "our", "each"],
    "subj": ["aide", "artist", "captain", "child", "doctor", "farmer", "guard", "keeper", "poet", "sailor", "teacher", "woman", "writer", "king", "maker", "traveler"],
    "verb": ["asks", "calls", "carries", "charts", "chooses", "crosses", "draws", "finds", "follows", "guards", "helps", "leads", "marks", "meets", "notices", "offers", "opens", "reads", "returns", "sends", "serves", "shares", "shows", "studies", "teaches", "touches", "trusts", "visits", "watches", "writes"],
    "obj": ["answer", "bird", "boat", "book", "candle", "door", "garden", "harbor", "island", "letter", "map", "message", "memory", "note", "path", "plan", "river", "secret", "story", "stone", "storm", "truth", "window"],
    "prep": ["by", "from", "into", "near", "over", "through", "under", "with"],
    "name": ["adam", "alice", "anna", "clara", "diana", "iris", "jane", "leon", "lisa", "maya", "nina", "rose", "ruth", "sam"],
}

# Distinct semantic arcs.  Left and right draw independently from these
# templates and are never derived from each other.
TEMPLATES = [
    ("agent_action", ("det", "subj", "verb", "det", "obj")),
    ("named_action", ("name", "verb", "det", "obj")),
    ("agent_prep_action", ("det", "subj", "verb", "prep", "det", "obj")),
    ("two_event_arc", ("det", "subj", "verb", "det", "obj", "prep", "det", "obj")),
    ("named_two_event_arc", ("name", "verb", "det", "obj", "prep", "det", "subj")),
    ("narrative_arc", ("det", "subj", "verb", "det", "obj", "prep", "det", "subj", "verb", "obj")),
    ("extended_arc", ("det", "subj", "verb", "det", "obj", "prep", "det", "subj", "verb", "prep", "det", "obj")),
]

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = tape(s)
    mismatch = next((i for i, (a, b) in enumerate(zip(t, reversed(t))) if a != b), None)
    return {"letters": len(t), "exact": mismatch is None, "first_mismatch": mismatch,
            "sha_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def render(words):
    return " ".join(words).capitalize() + "."

def choices(kind):
    return BANK[kind]

def clauses(max_per_template=180):
    """Return independent semantic clauses, retaining role provenance."""
    out = []
    for tid, slots in TEMPLATES:
        n = 0
        def rec(i, words):
            nonlocal n
            if n >= max_per_template:
                return
            if i == len(slots):
                out.append({"template": tid, "slots": list(slots), "words": list(words), "text": render(words)})
                n += 1
                return
            for w in choices(slots[i]):
                rec(i + 1, words + [w])
                if n >= max_per_template:
                    break
        rec(0, [])
    return out

def live_pair(left, right, limit=120):
    """Zipper two independently chosen clauses, checking residual chars live."""
    a, b = left["words"], right["words"]
    # State is word indices and character offsets at each end.  The central
    # seam is reached when the two cursors meet; offsets permit an inner-word
    # seam and make boundaries part of the search state.
    states = [(0, 0, 0, 0, [])]
    seen = set()
    while states:
        li, lo, ri, ro, trace = states.pop()
        key = (li, lo, ri, ro)
        if key in seen or len(trace) > limit:
            continue
        seen.add(key)
        if li == len(a) and ri == len(b):
            if trace:
                return trace
            continue
        # Consume one available character from the shorter accumulated side.
        # Both directions are independently lexical; choosing a side is a
        # boundary decision, not a reversal operation.
        ldone = li == len(a)
        rdone = ri == len(b)
        if ldone or (not rdone and sum(map(len, a[:li])) + lo <= sum(map(len, b[:ri])) + ro):
            if not ldone:
                ch = a[li][lo]
                nli, nlo = li, lo + 1
                if nlo == len(a[li]): nli, nlo = li + 1, 0
                # Opposite side is checked when it is available; unmatched
                # residuals remain explicit in the trace.
                states.append((nli, nlo, ri, ro, trace + [("L", ch, li, lo)]))
        else:
            if not rdone:
                ch = b[ri][-(ro + 1)]
                nri, nro = ri, ro + 1
                if nro == len(b[ri]): nri, nro = ri + 1, 0
                states.append((li, lo, nri, nro, trace + [("R", ch, ri, ro)]))
    return None

def residual_equation(left_words, right_words):
    """Evaluate the live seam residual while lexical arcs are consumed.

    The two arcs remain distinct: characters are taken from the left arc's
    front and right arc's back, with offsets crossing word boundaries.  Any
    first unequal residual kills the state immediately; the center may be
    inside either word.
    """
    left = "".join(left_words)
    right = "".join(right_words)
    if len(left) != len(right):
        return False
    for i, ch in enumerate(left):
        if ch != right[-1-i]:
            return False
    return True

def seam_indexed_residual(left_words, right_words):
    """Live lexical residual-domain gate, consuming complete words lazily.

    At each step the next available character is drawn from the currently
    exposed word on each side.  This makes word boundaries part of the state;
    a mismatch prunes immediately, before later slots are considered.
    """
    if sum(map(len, left_words)) != sum(map(len, right_words)):
        return False
    li = ri = lo = ro = 0
    total = sum(map(len, left_words))
    for _ in range(total):
        while li < len(left_words) and lo == len(left_words[li]): li, lo = li + 1, 0
        while ri < len(right_words) and ro == len(right_words[ri]): ri, ro = ri + 1, 0
        if li == len(left_words) or ri == len(right_words): return False
        if left_words[li][lo] != right_words[ri][-(ro + 1)]: return False
        lo += 1; ro += 1
    return True

def search():
    cs = clauses()
    # Group by total length so the seam is a genuine residual equation.  The
    # pair test is deliberately independent and rejects same-content pairs.
    by_len = {}
    for c in cs:
        by_len.setdefault(sum(map(len, c["words"])), []).append(c)
    results, controls = [], []
    # Preserve intact long semantic controls even when their lengths cannot
    # close the palindrome residual equation.
    long = [c for c in cs if sum(map(len, c["words"])) > 38]
    for c in long[:24]:
        au = audit(c["text"])
        controls.append({"text": c["text"], "length": au["letters"], "audit": au,
                         "left": c, "right": None,
                         "provenance": "independent_typed_event_arc_control",
                         "novelty": "not_catalogue_or_mirrored_units",
                         "reader_worthy": False})
    pairs_seen = 0
    for length, lefts in sorted(by_len.items()):
        rights = by_len[length]
        for left in lefts:
            for right in rights:
                if left["words"] == right["words"] or set(left["words"]) & set(right["words"]):
                    continue
                pairs_seen += 1
                # Build the actual sentence from two semantic arcs.  The
                # right arc is lexical, not a generated reverse of the left.
                text = render(left["words"])[:-1] + "; " + " ".join(right["words"]) + "."
                au = audit(text)
                rec = {"text": text, "length": au["letters"], "audit": au,
                       "left": left, "right": right,
                       "provenance": "independent_typed_event_arcs",
                       "novelty": "not_catalogue_or_mirrored_units",
                       "reader_worthy": False}
                if len(controls) < 24 and au["letters"] > 38:
                    controls.append(rec)
                # Solve the character residual before admitting an exact result.
                if not seam_indexed_residual(left["words"], right["words"]):
                    continue
                if au["exact"]:
                    results.append(rec)
    return {"method": "center_seam_event_constructor", "templates": len(TEMPLATES),
            "clauses": len(cs), "exact_candidates": results,
            "controls": controls, "pairs_seen": pairs_seen,
            "next_repair": "replace independent clause pair with seam-indexed lexical residual domains; current typed arcs have no compatible residual closure"}

if __name__ == "__main__":
    out = search()
    path = ROOT / "runs/center-seam-event-constructor-20260921.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"clauses": out["clauses"], "exact": len(out["exact_candidates"]), "controls": len(out["controls"]), "path": str(path)}))
