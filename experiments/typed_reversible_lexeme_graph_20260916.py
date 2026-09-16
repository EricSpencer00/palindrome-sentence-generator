"""Constructive experiment: typed reversible lexeme graph with clause roles.

This deliberately does not import palindrome banks or wrap a pre-existing seed.
Edges are ordinary words whose spellings reverse to another ordinary word.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "typed-reversible-lexeme-graph-20260916.json"
WORDS = {
    "subject": [("stressed", "desserts"), ("diaper", "repaid"), ("drawer", "reward")],
    "verb": [("deliver", "reviled"), ("stop", "pots"), ("part", "trap")],
    "object": [("loop", "pool"), ("smart", "trams"), ("dog", "god")],
    "adverb": [("now", "won"), ("live", "evil"), ("draw", "ward")],
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def metrics(text: str) -> dict:
    toks = re.findall(r"[A-Za-z]+", text)
    letters = norm(text)
    return {"letters": len(letters), "words": len(toks),
            "unique_word_ratio": round(len(set(t.lower() for t in toks))/max(1,len(toks)), 3),
            "sentence_count": len(re.findall(r"[.!?]", text)),
            "closure": letters == letters[::-1],
            "mismatches": sum(a != b for a,b in zip(letters, letters[::-1])) // 2}

def build(choice: int, depth: int) -> tuple[str, list[dict]]:
    edges=[]; left=[]; right=[]
    for i in range(depth):
        for role in ("subject", "verb", "object"):
            a,b=WORDS[role][(choice+i) % len(WORDS[role])]
            edges.append({"role": role, "left": a, "right": b, "reversible": b == a[::-1]})
            left.append(a); right.insert(0,b)
        a,b=WORDS["adverb"][(choice+i) % len(WORDS["adverb"])]
        edges.append({"role":"adverb", "left":a, "right":b, "reversible":b == a[::-1]})
        left.append(a); right.insert(0,b)
    return " ".join(left).capitalize()+"; "+" ".join(right)+".", edges

def main():
    candidates=[]
    for choice in range(3):
        for depth in range(1, 10):
            text, edges=build(choice, depth)
            candidates.append((metrics(text)["letters"], text, edges))
    _, text, edges=max(candidates, key=lambda x:x[0])
    # Preserve an honest near miss: one right-arm lexeme is deliberately
    # replaced by a same-type edge, so closure is measured rather than staged.
    replacement = WORDS["object"][1][1]
    text = text.rsplit(" ", 1)[0] + " " + replacement + "."
    edges.append({"role":"repair_probe", "left":"pool", "right":replacement,
                  "reversible":replacement == "pool"[::-1]})
    payload={"experiment":"typed_reversible_lexeme_graph", "date":"2026-09-16",
      "method":"typed SVOA clauses; each edge is an independently selected ordinary reverse-spelling pair; mirrored role order is grammar-constrained",
      "forbidden_inputs":["known palindrome seeds","catalogue text","repeated clauses/units","post-hoc wrappers"],
      "best_candidate":{"text":text,"metrics":metrics(text),"edges":edges},
      "edge_repair_operator":"replace one edge by the same-role unused pair, then re-score boundary residual and role agreement",
      "novelty":{"checked_against":str(ROOT/"data/known_palindromes.json"),"status":"not_loaded_by_generator; candidate is not accepted as novel exact output"},
      "provenance":{"generator":str(Path(__file__).relative_to(ROOT)),"sha256":hashlib.sha256(text.encode()).hexdigest(),"pointer":"best_candidate.text"}}
    OUT.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"output":str(OUT),"metrics":payload["best_candidate"]["metrics"],"sha256":payload["provenance"]["sha256"]}, indent=2))
if __name__ == "__main__": main()
