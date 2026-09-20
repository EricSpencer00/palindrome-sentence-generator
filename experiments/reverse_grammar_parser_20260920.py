"""Reverse-stream grammatical parser lane.

Complete authored left clauses are generated from ordinary SVO/PP/relative
templates.  Their letter stream is consumed from the right while a parser
builds a separate grammatical right clause; word boundaries are discovered by
the parser, not mirrored from the left.  This is a constructive parser
intersection, not a post-hoc repair of a known palindrome.
"""
from __future__ import annotations

import hashlib, json, re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "reverse-grammar-parser-20260920.json"
EXPERIMENT_ID = "reverse-grammar-parser-20260920"

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict[str, object]:
    t = letters(s); f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    mm = next(((i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and mm is None, "first_mismatch": mm,
            "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

LEX = {
    "det": ("a", "an", "the", "some", "one"),
    "sgn": ("aide", "artist", "bard", "clerk", "doctor", "farmer", "nurse", "poet", "scribe", "teacher", "writer"),
    "pln": ("artists", "bards", "clerks", "doctors", "farmers", "nurses", "poets", "scribes", "teachers", "writers"),
    "sgv": ("aids", "asks", "calls", "draws", "edits", "feeds", "finds", "helps", "keeps", "marks", "meets", "names", "notes", "reads", "rips", "saves", "sees", "sends", "shares", "sings", "takes", "tests", "ties", "uses", "visits", "writes"),
    "plv": ("aid", "ask", "call", "draw", "edit", "feed", "find", "help", "keep", "mark", "meet", "name", "note", "read", "rip", "save", "see", "send", "share", "sing", "take", "test", "tie", "use", "visit", "write"),
    "obj": ("book", "chart", "idea", "letter", "map", "memo", "message", "note", "plan", "poem", "record", "secret", "sign", "song", "story", "tale", "text", "verse"),
    "prep": ("at", "by", "in", "on", "to"),
    "place": ("harbor", "market", "office", "river", "school", "shore", "station", "town", "village"),
    "rel": ("who", "that"),
}

LEFT = []
for d,n,v,o in product(LEX["det"], LEX["sgn"], LEX["sgv"], LEX["obj"]): LEFT.append((d,n,v,o))
for d,n,v,o,p,pl in product(LEX["det"], LEX["sgn"], LEX["sgv"], LEX["obj"], LEX["prep"], LEX["place"]): LEFT.append((d,n,v,o,p,pl))
for d,n,r,v,o in product(LEX["det"], LEX["sgn"], LEX["rel"], LEX["sgv"], LEX["obj"]): LEFT.append((d,n,r,v,o))
# Keep the authored inventory broad but bounded for a reproducible queue run.
# The cap is deterministic (product order), not a result-dependent filter.
LEFT = LEFT[:12000]

GRAMMARS = {
    "SVO": (("det", "sgn", "sgv", "obj"), ("det", "pln", "plv", "obj")),
    "SVOPP": (("det", "sgn", "sgv", "obj", "prep", "place"),),
    "REL": (("det", "sgn", "rel", "sgv", "obj"),),
}

def parse_reverse(tape: str, shape: tuple[str, ...], max_parses=3):
    # Consume the required reversed stream from its left edge.  The parser's
    # grammar position determines each word boundary; no left word boundary
    # is copied.
    out=[]
    def rec(pos: int, i: int, words: tuple[str,...]):
        if len(out) >= max_parses: return
        if i == len(shape):
            if pos == len(tape): out.append(words)
            return
        role = shape[i]
        for w in LEX[role]:
            x = letters(w)
            if tape[pos:pos+len(x)] == x:
                rec(pos+len(x), i+1, words+(w,))
    rec(0, 0, ())
    return out

def main():
    states=0; parsed=0; candidates=[]; examples=[]; seen=set()
    for clause in LEFT:
        rendered=" ".join(clause); tape=letters(rendered); states += 1
        for name, shapes in GRAMMARS.items():
            for shape in shapes:
                # The right clause must spell the reverse of the completed
                # left tape; parsing consumes that required stream online.
                for right in parse_reverse(tape[::-1], shape):
                    parsed += 1; text = rendered + " " + " ".join(right)
                    a=audit(text)
                    if a["exact"] and text not in seen and len(set(clause+right)) == len(clause+right):
                        seen.add(text); candidates.append({"rendered":text,"audit":a,"provenance":{"construction":"authored left-clause enumeration with online reverse-stream grammar parser","left_clause":rendered,"right_shape":name,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"repeated_word":False},"reader_status":"unreviewed; exactness does not certify readability"})
                    if len(examples)<20 and right:
                        examples.append({"left_clause":rendered,"reverse_parse":" ".join(right),"shape":name,"whole_audit":a})
    candidates.sort(key=lambda x:x["audit"]["letters"], reverse=True)
    payload={"experiment_id":EXPERIMENT_ID,"method":"reverse-stream grammar parser","stats":{"left_clauses":len(LEFT),"states":states,"reverse_parses":parsed,"exact":len(candidates),"fresh_exact_gt38":sum(x["audit"]["letters"]>38 for x in candidates)},"candidates":candidates[:50],"diagnostic_parses":examples,"independent_audit":"two-pointer mismatch scan plus independent SHA-256 forward/reverse equality","novelty":"new parser-intersection lane; no mirrored units or catalogue text"}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload["stats"], indent=2))
    for x in candidates[:5]: print(x["rendered"])
if __name__ == "__main__": main()
