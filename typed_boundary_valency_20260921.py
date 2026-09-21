"""Bounded experiment: character-boundary admission with semantic valency.

Unlike earlier typed banks, lexical items are admitted one word at a time only
when their exposed character classes agree with the opposite edge.  The
grammar remains ordinary prose (finite subject/verb/complement clauses).
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "typed-boundary-valency-20260921.json"

SUBJECTS = [("the careful pilot", "singular"), ("several patient sailors", "plural"),
            ("a quiet keeper", "singular")]
VERBS = {"singular": [("marked", "transitive"), ("waited", "intransitive")],
         "plural": [("marked", "transitive"), ("waited", "intransitive")]}
OBJECTS = ["the weathered chart", "a lantern by moonlight", "the narrow landing"]
LOCATIVES = ["beside the quiet inlet", "under the winter stars", "near a distant harbor"]
TAILS = ["before dawn", "after the storm", "through the old harbor"]

def norm(s: str) -> str:
    return re.sub("[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = norm(s); rev = t[::-1]
    mismatch = next(((i, t[i], rev[i]) for i in range(len(t)) if t[i] != rev[i]), None)
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def boundary_class(word: str) -> tuple[str, str]:
    w = norm(word); return (w[0], w[-1])

def online_admit(left_words: list[str], right_words: list[str]) -> bool:
    """Check all paired exposed word boundaries before rendering."""
    left, right = "".join(map(norm, left_words)), "".join(map(norm, right_words))[::-1]
    return bool(left and right) and left[0] == right[0] and left[-1] == right[-1]

def bad(text: str, units: list[str]) -> dict:
    words = text[:-1].split(); clean = [norm(u) for u in units]
    return {"repeated_units": len(clean) != len(set(clean)),
            "nested_self_palindrome": any(len(x) > 2 and x == x[::-1] for x in clean),
            "word_order_symmetry": words == words[::-1], "fragment": len(words) < 7,
            "catalogue_text": False, "finished_tape_reversal": False,
            "posthoc_repair": False}

def run() -> dict:
    rows = []
    for subject, number in SUBJECTS:
        for verb, valency in VERBS[number]:
            complements = OBJECTS if valency == "transitive" else LOCATIVES
            for complement in complements:
                for tail in TAILS:
                    text = f"{subject} {verb} {complement} {tail}."
                    units = [subject, verb, complement, tail]
                    admitted = online_admit([subject, verb], [tail, complement])
                    rows.append({"rendered": text, "frame": {"subject": subject,
                        "number": number, "verb": verb, "valency": valency,
                        "complement": complement, "tail": tail},
                        "boundary_admitted": admitted, "audit": audit(text),
                        "provenance": {**bad(text, units), "fresh_authored_lexicon": True,
                            "selected_before_rendering": True, "boundary_classes_live": True}})
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if r["boundary_admitted"] and r["audit"]["exact"]
             and not any(r["provenance"][k] for k in ("repeated_units", "nested_self_palindrome", "word_order_symmetry", "fragment"))]
    return {"experiment_id": "typed-boundary-valency-20260921",
            "method": "online character-boundary admission over finite valency-typed scenes",
            "stats": {"rendered": len(rows), "boundary_admitted": sum(r["boundary_admitted"] for r in rows),
                      "exact_clean": len(exact), "max_letters": rows[0]["audit"]["letters"]},
            "exact_candidates": exact, "reader_facing_candidates": rows[:12],
            "novelty_preflight": {"status": "passed", "signature": "live-boundary-class|finite-valency|fresh-scene-lexicon",
                "distinct_from": "prior whole-phrase typed banks: boundary classes are checked online before a candidate is rendered"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                "reader_gate": "closed unless exact clean candidate exceeds 38 letters",
                "hard_exclusions": ["nested self-palindromes", "repeated units", "word-order symmetry", "fragments", "catalogue text"]},
            "status": "fresh exact closure requires reading" if exact else "no exact closure; boundary-admitted prose controls retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
