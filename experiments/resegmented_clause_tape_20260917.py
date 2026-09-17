"""Exact-tape resegmentation with independent clause grammars.

The left side is generated as intact authored clauses.  Its reversed character
tape is then segmented by a separate right-side grammar; no mirrored words or
catalogue strings are supplied to either side.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LEFT = {
    "subject": ("aide", "artist", "child", "dancer", "friend", "judge", "nurse", "poet", "sailor", "teacher"),
    "verb": ("admires", "answers", "carries", "draws", "guides", "keeps", "names", "opens", "reads", "sees"),
    "object": ("a map", "a note", "a poem", "a song", "the book", "the gate", "the river", "new art", "old letters", "quiet music"),
}
RIGHT = {
    "det": ("a", "an", "the", "new", "old", "one", "some"),
    "noun": ("artist", "bird", "child", "door", "friend", "garden", "judge", "letter", "map", "note", "poem", "river", "song", "teacher", "word"),
    "verb": ("admire", "answers", "carries", "draws", "guides", "keeps", "names", "opens", "reads", "sees", "writes"),
    "prep": ("in", "on", "near", "over", "under"),
}

def tape(s: str) -> str:
    return re.sub("[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = tape(s)
    return {"exact": bool(t) and t == t[::-1], "letters": len(t),
            "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "forward": t, "reverse": t[::-1]}

def left_clauses():
    # Two ordinary clause shapes, with boundaries retained as provenance.
    for s, v, o in itertools.product(LEFT["subject"], LEFT["verb"], LEFT["object"]):
        yield (f"{s} {v} {o}", ("subject", "verb", "object"))
    for s, v, o in itertools.product(LEFT["subject"], LEFT["verb"], LEFT["object"]):
        yield (f"the {s} {v} {o}", ("det_subject", "subject", "verb", "object"))

def right_segmentations(target: str, limit=20):
    # Independent grammar: determiner+noun, optionally verb+noun; no left words
    # are consulted when choosing boundaries.
    words = {tape(w): w for vals in RIGHT.values() for w in vals}
    grammar = [("det", "noun"), ("det", "noun", "verb"),
               ("det", "noun", "verb", "det", "noun"),
               ("noun", "verb", "det", "noun")]
    byslot = {"det": RIGHT["det"], "noun": RIGHT["noun"], "verb": RIGHT["verb"]}
    out=[]
    def rec(pos, slots, chosen):
        if len(out) >= limit: return
        if not slots:
            if pos == len(target): out.append(tuple(chosen))
            return
        slot=slots[0]
        for w in byslot[slot]:
            z=tape(w)
            if target.startswith(z, pos): rec(pos+len(z), slots[1:], chosen+[w])
    for shape in grammar: rec(0, shape, [])
    return out

def independent_check(rendered):
    # Deliberately separate implementation from audit().
    q="".join(c for c in rendered.casefold() if "a" <= c <= "z")
    return q == "".join(reversed(q)) and len(q)>0

def main():
    checked=0; exact=[]; near=[]
    for phrase, shape in left_clauses():
        checked += 1
        target=tape(phrase)[::-1]
        parses=right_segmentations(target)
        if parses:
            for p in parses:
                rendered=phrase+"; "+" ".join(p)
                a=audit(rendered)
                if a["exact"] and independent_check(rendered):
                    exact.append({"text": rendered, "left_shape": shape, "right_words": p, "audit": a,
                                  "provenance":"authored-left/resegmented-right"})
        # Keep a concrete diagnostic: longest prefix accepted by any RHS grammar.
        best=0
        for n in range(len(target), 0, -1):
            if right_segmentations(target[:n], limit=1): best=n; break
        if best and len(near)<12: near.append({"left":phrase,"reverse_prefix_letters":best,"target_letters":len(target),"next_repair":"add a typed right-side clause word at the first unmatched seam"})
    result={"method":"independent authored-clause tape resegmentation","checked_left_clauses":checked,
            "exact_count":len(exact),"exact":exact,"near_misses":near,
            "validator":"independent_check plus audit; no catalogue fixture or mirrored word bank"}
    path=ROOT/"runs/resegmented-clause-tape-20260917.json"; path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"checked":checked,"exact":len(exact),"near_misses":len(near),"artifact":str(path)},indent=2))
    for row in exact[:5]: print(row["text"], row["audit"])

if __name__ == "__main__": main()
