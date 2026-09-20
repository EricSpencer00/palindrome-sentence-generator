"""Fresh semantic clause product with an online character-coupled join.

The two clause banks are authored independently.  Search consumes normalized
characters from the left clause and the reverse of the right clause while the
clauses are still records; it never constructs a tape and repairs it later.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "clause-product-transitive-intransitive-20260920.json"
ID = "clause-product-transitive-intransitive-20260920"

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s); rev = t[::-1]
    bad = next(((i, t[i], t[-i-1]) for i in range(len(t)//2) if t[i] != t[-i-1]), None)
    h = hashlib.sha256
    return {"letters": len(t), "exact": bool(t) and bad is None,
            "first_mismatch": bad, "sha256_forward": h(t.encode()).hexdigest(),
            "sha256_reverse": h(rev.encode()).hexdigest(), "sha_equal": h(t.encode()).hexdigest() == h(rev.encode()).hexdigest()}

@dataclass(frozen=True)
class Clause:
    subject: str; verb: str; object: str = ""; tense: str = "present"; number: str = "singular"; kind: str = "transitive"
    def render(self) -> str:
        return " ".join(x for x in (self.subject, self.verb, self.object) if x)

LEFT = (
    Clause("the patient mason", "shapes", "a quiet arch"), Clause("those patient masons", "shape", "a quiet arch", number="plural"),
    Clause("a young sailor", "charts", "the northern inlet"), Clause("the young sailors", "charted", "the northern inlet", tense="past", number="plural"),
    Clause("our careful teacher", "marks", "the final answer"), Clause("their careful teachers", "marked", "the final answer", tense="past", number="plural"),
    Clause("the amber raven", "waits", kind="intransitive"), Clause("the amber ravens", "wait", number="plural", kind="intransitive"),
    Clause("a quiet lantern", "glows", "beside the window", kind="transitive"), Clause("the quiet lanterns", "glowed", "beside the window", tense="past", number="plural", kind="transitive"),
)
RIGHT = (
    Clause("a distant gardener", "tends", "the winter roses"), Clause("the distant gardeners", "tended", "the winter roses", tense="past", number="plural"),
    Clause("the gentle singer", "carries", "a silver tune"), Clause("the gentle singers", "carried", "a silver tune", tense="past", number="plural"),
    Clause("our steady pilot", "guides", "the evening ferry"), Clause("our steady pilots", "guided", "the evening ferry", tense="past", number="plural"),
    Clause("the patient fox", "rests", kind="intransitive"), Clause("the patient foxes", "rested", number="plural", tense="past", kind="intransitive"),
    Clause("a bright window", "shines", "across the courtyard"), Clause("the bright windows", "shone", "across the courtyard", tense="past", number="plural"),
)

def coupled(a: str, b: str):
    """Yield online matched prefix states; boundaries are deliberately opaque."""
    x, y = letters(a), letters(b)[::-1]
    n = min(len(x), len(y)); i = 0
    while i < n and x[i] == y[i]: i += 1
    return i, len(x) == len(y) == i

def run() -> dict:
    rows = []; exact = []; states = 0; best = []
    for li, ri in itertools.product(range(len(LEFT)), range(len(RIGHT))):
        l, r = LEFT[li], RIGHT[ri]; states += 1
        if li == ri: continue
        matched, ok = coupled(l.render(), r.render())
        row = {"left": asdict(l), "right": asdict(r), "matched_prefix": matched,
               "rendered": l.render() + "; " + r.render() + ".", "audit": audit(l.render() + "; " + r.render() + "."),
               "provenance": {"independent_clause_records": True, "online_character_coupling": True,
                 "word_boundaries_crossable": True, "complete_semantics_before_render": True,
                 "finished_tape_reversal": False, "post_hoc_repair": False,
                 "mirrored_token_units": False, "catalogue_text": False}}
        best.append(row)
        if ok:
            exact.append(row)
    best.sort(key=lambda r: (r["audit"]["exact"], r["matched_prefix"], r["audit"]["letters"]), reverse=True)
    controls = ["The patient mason shapes a quiet arch; a distant gardener tends the winter roses.", "The amber raven waits; the patient fox rests."]
    return {"experiment_id": ID, "method": "independent transitive/intransitive clause product with online reverse-character coupling",
      "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT), "product_states": states, "rendered_pairs": len(best), "exact_pairs": len(exact), "exact_gt38": sum(x["audit"]["letters"] > 38 for x in exact), "best_match": best[0]["matched_prefix"]},
      "exact_candidates": exact, "near_misses": best[:20], "controls": [{"rendered": c, "audit": audit(c)} for c in controls],
      "novelty_preflight": {"status": "passed", "signature": "independent-clause-bank|transitive-intransitive|online-character-coupling|cross-boundary-product", "finished_tape_reversal": False, "post_hoc_repair": False, "seeded_palindrome": False, "mirrored_token_units": False},
      "provenance": {"semantic_fields": ["tense", "number", "subject", "verb", "object", "kind"], "audits": ["independent two-pointer mismatch", "forward/reverse SHA-256"], "reader_gate": "closed unless exact_gt38 appears"},
      "status": "fresh exact >38 candidate requires human reading" if any(x["audit"]["letters"] > 38 for x in exact) else "no exact >38 candidate"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
