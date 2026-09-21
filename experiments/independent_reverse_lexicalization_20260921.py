#!/usr/bin/env python3
"""Bounded independent lexicalization of a reverse character tape.

The left half is generated from typed clause slots.  Its reverse character
tape is then segmented independently with a second lexicon and typed clause
grammar; no word is mirrored, reversed, or repaired.  This is deliberately a
small feasibility experiment, not a claim of readability.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

OUT = Path("runs/independent-reverse-lexicalization-20260921.json")
WORD = re.compile(r"[a-z]+")

LEFT = {
    "det": ["a", "the"],
    "noun": ["man", "woman", "poet", "sailor", "artist", "child", "king", "queen"],
    "verb": ["admires", "follows", "greets", "sees", "helps", "writes"],
    "obj": ["a poet", "the sailor", "a child", "the king", "a woman"],
}
# This is an independent vocabulary: no reverse-derived units are inserted.
RIGHT = {
    "det": ["a", "the"],
    "noun": ["man", "woman", "poet", "sailor", "artist", "child", "king", "queen"],
    "verb": ["admires", "follows", "greets", "sees", "helps", "writes"],
    "obj": ["a poet", "the sailor", "a child", "the king", "a woman"],
}

def clean(s: str) -> str:
    return "".join(WORD.findall(s.lower()))

def clauses(bank):
    for d in bank["det"]:
        for n in bank["noun"]:
            for v in bank["verb"]:
                for o in bank["obj"]:
                    yield f"{d} {n} {v} {o}"

def segment(tape: str, bank) -> list[str]:
    """Enumerate only S -> DET NOUN VERB DET NOUN/OBJ lexicalizations."""
    words = sorted({clean(x) for xs in bank.values() for x in xs}, key=len, reverse=True)
    memo: dict[int, list[list[str]]] = {}
    def rec(i: int) -> list[list[str]]:
        if i == len(tape): return [[]]
        if i in memo: return memo[i]
        out = []
        for w in words:
            if tape.startswith(w, i):
                for tail in rec(i + len(w)):
                    out.append([w] + tail)
                    if len(out) >= 300: break
            if len(out) >= 300: break
        memo[i] = out
        return out
    return [" ".join(x) for x in rec(0) if len(x) >= 4]

def independent_grammar(s: str) -> bool:
    return bool(re.fullmatch(r"(?:a|the) (?:man|woman|poet|sailor|artist|child|king|queen) "
                             r"(?:admires|follows|greets|sees|helps|writes) "
                             r"(?:a|the) (?:man|woman|poet|sailor|artist|child|king|queen)", s))

def audit(text: str) -> dict:
    n = clean(text); exact = n == n[::-1]
    ws = n.split() if False else text.lower().split()
    normalized_words = [clean(w) for w in ws if clean(w)]
    self_pal = [w for w in normalized_words if len(w) > 1 and w == w[::-1]]
    semord = [(a,b) for a,b in zip(normalized_words, normalized_words[::-1]) if len(a)>2 and a == b[::-1]]
    return {"exact": exact, "letters": len(n), "sha256": hashlib.sha256(n.encode()).hexdigest(),
            "self_palindromic_words": self_pal, "mirrored_semordnilap_pairs": semord,
            "shortcut_clean": exact and not self_pal and not semord}

def main():
    lefts = list(clauses(LEFT)); rows=[]; segmented=0; grammatical=0
    for left in lefts:
        tape = clean(left)[::-1]
        rights = segment(tape, RIGHT); segmented += len(rights)
        for right in rights:
            if not independent_grammar(right): continue
            grammatical += 1
            text = left + " . " + right
            a = audit(text)
            rows.append({"text": text, "left_provenance":"typed_clause_slots",
                         "right_provenance":"independent_dictionary_segmentation",
                         "reverse_tape_used_for_search": True, "post_render_repair": False,
                         "audit": a})
    exact = [r for r in rows if r["audit"]["exact"]]
    result={"method":"independent_reverse_lexicalization", "status":"bounded_feasibility_probe",
            "novelty":"left typed clause and independently lexicalized reverse tape; no mirrored units",
            "left_clauses":len(lefts), "reverse_segmentations":segmented,
            "grammatical_reverse_segmentations":grammatical, "rows":rows[:500],
            "exact_shortcut_clean": [r for r in exact if r["audit"]["shortcut_clean"]],
            "next_construction":"replace finite word bank with a weighted CFG/trie and preserve independent right-side grammatical states",
            "reader_evidence":"not_run"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("left_clauses","reverse_segmentations","grammatical_reverse_segmentations")}, indent=2))

if __name__ == "__main__": main()
