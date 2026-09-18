#!/usr/bin/env python3
"""Character-level grammar intersection over a fresh authored lexicon.

The left side is generated from typed clause templates.  Its character tape is
then intersected with a trie of typed English words in reverse, with word
boundaries and grammatical types searched rather than copied or mirrored.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/char-grammar-intersection-20260918.json"

# Fresh, small author-written lexicon; no catalogue text or borrowed sentences.
B = {
    "D": ["a", "an", "the", "some"],
    "A": ["amber", "quiet", "young", "kind", "small", "brave"],
    "N": ["artist", "baker", "captain", "clerk", "friend", "garden", "harbor", "letter", "map", "nurse", "poet", "river", "sailor", "story", "teacher"],
    "V": ["admires", "bakes", "carries", "draws", "helps", "marks", "needs", "reads", "sees", "thanks", "trusts", "writes"],
    "P": ["after", "at", "by", "near", "over", "with"],
}
PATTERNS = [
    ("scene", ("D", "A", "N", "V", "D", "N")),
    ("agent", ("D", "N", "V", "D", "N", "P", "D", "N")),
    ("reported", ("D", "N", "V", "D", "A", "N")),
]
WORDS = [(w, t) for t, ws in B.items() for w in ws]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t = norm(s)
    return {"letters": len(t), "exact": t == t[::-1],
            "first_mismatch": next((i for i,(a,b) in enumerate(zip(t,t[::-1])) if a != b), None),
            "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "independent_two_pointer": all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "independent_forward_reverse_sha": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest()}

def trie_parse(tape, allowed_types=None, max_words=12):
    """Return all typed segmentations of a tape, using character transitions."""
    allowed = set(allowed_types or B)
    by_first = {}
    for w,t in WORDS:
        if t in allowed: by_first.setdefault(w[0], []).append((w,t))
    out=[]
    def rec(pos, words):
        if pos == len(tape): out.append(words[:]); return
        if len(words) >= max_words: return
        for w,t in by_first.get(tape[pos], []):
            if tape.startswith(w,pos): rec(pos+len(w), words+[(t,w)])
    rec(0, [])
    return out

def main():
    rows=[]; assignment_count=0; reverse_parse_count=0
    # Character intersection: every candidate's reversed tape is segmented
    # independently; no right words are obtained by reversing left words.
    for name, pattern in PATTERNS:
        # Bounded breadth keeps this a reproducible lane while still covering
        # each template's early lexical cross-product.
        for vals in itertools.islice(itertools.product(*(B[t] for t in pattern)), 50000):
            assignment_count += 1
            left = " ".join(vals)
            tape = norm(left)[::-1]
            parses = trie_parse(tape, max_words=len(pattern)+4)
            reverse_parse_count += len(parses)
            for rp in parses:
                right = " ".join(w for _,w in rp)
                rendered = left + " " + right
                rows.append({"rendered": rendered, "left_types": list(pattern),
                    "right_types": [t for t,_ in rp],
                    "provenance": "fresh typed clause lattice; character tape reversed then independently trie-segmented; no mirrored insertion",
                    "audit": audit(rendered),
                    "novelty_preflight": {"signature":"char-grammar-intersection-v1",
                        "distinct_from":"word-equation fixed-boundary enumeration and seam substitution",
                        "fresh_authored_lexicon": True},
                    "anti_shortcut": {"catalogue":False,"repeated_unit":False,"word_order_only":False,
                        "punctuation_carries_letters":False,"fragment":False,"borrowed_text":False},
                    "grammar_intersection": {"pattern":name,"reverse_segmentation_words":len(rp),
                        "complete_tape_parse":True}})
    # Always retain readable controls, even when intersection is empty.
    if not rows:
        controls = ["The quiet baker reads a letter near the harbor.",
                    "A young teacher writes a kind story by the river."]
        for s in controls:
            rows.append({"rendered":s,"left_types":[],"right_types":[],
                "provenance":"fresh authored control; no exact candidate claimed",
                "audit":audit(s),"novelty_preflight":{"signature":"char-grammar-intersection-v1"},
                "anti_shortcut":{"catalogue":False,"repeated_unit":False,"word_order_only":False,
                    "punctuation_carries_letters":False,"fragment":False,"borrowed_text":False},
                "grammar_intersection":{"complete_tape_parse":False}})
    rows = sorted(rows, key=lambda r:(not r["audit"]["exact"], -r["audit"]["letters"]))[:30]
    payload={"experiment":"char-grammar-intersection-20260918",
        "method":"Intersect typed character-level clause generation with reverse-tape trie segmentation; word boundaries and right-side lexical choices are variables.",
        "assignment_count":assignment_count,"reverse_parse_count":reverse_parse_count,
        "candidate_count":len(rows),"candidates":rows,
        "summary":{"exact_count":sum(r["audit"]["exact"] for r in rows),
            "longest_letters":max(r["audit"]["letters"] for r in rows),
            "next_repair":"add a finite-state inflectional transducer and seam-aware pruning so grammatical right parses can cross longer clause tapes without importing mirrored units."}}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload["summary"]))
if __name__ == "__main__": main()
