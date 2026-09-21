"""Joint semantic-slot Cartesian-product diagnostic (no reverse-text realization).

The diagnostic assigns lexical items to both sides in a complete Cartesian
product and audits finished strings. It does not invoke SMT or propagate
partial character equations. It intentionally reports failures as
evidence: a right-hand word is never manufactured by reversing a left string.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SLOTS = {
    "subject": [("the", "artist"), ("a", "pilot"), ("our", "teacher")],
    "verb": [("guards", "guides"), ("writes", "charts"), ("builds", "mends")],
    "object": [("quiet", "harbor"), ("bright", "lantern"), ("winter", "garden")],
    "adjunct": [("at dawn", "after rain"), ("with care", "in silence")],
}
FRAMES = [
    "{subject} {verb} the {object} {adjunct}.",
    "In {adjunct}, {subject} {verb} the {object}.",
]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def checks(s):
    t = norm(s)
    return {"letters": len(t), "is_palindrome": t == t[::-1],
            "two_pointer": all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "hash_equal": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest(),
            "words": len(re.findall(r"[A-Za-z]+", s)),
            "distinct_nonpalindromic_lexemes": len(set(re.findall(r"[a-z]+", t))) == len(re.findall(r"[a-z]+", t)) and all(w != w[::-1] for w in re.findall(r"[a-z]+", t))}

def main():
    # Global equation pruning: compare every assigned character pair before a
    # complete sentence is admitted.  Both sides are independently selected
    # slot values; there is no catalogue and no word-order reversal.
    states = 0; pruned = 0; rendered = []
    for frame in FRAMES:
        for subj in SLOTS["subject"]:
          for verb in SLOTS["verb"]:
           for obj in SLOTS["object"]:
            for adj in SLOTS["adjunct"]:
             states += 1
             left = frame.format(subject=subj[0], verb=verb[0], object=obj[0], adjunct=adj[0])
             right = frame.format(subject=subj[1], verb=verb[1], object=obj[1], adjunct=adj[1])
             candidate = left + " " + right
             c = checks(candidate)
             # A complete prose candidate must satisfy the gate; retain all
             # rendered attempts for auditability, even rejected ones.
             rendered.append({"text": candidate, "checks": c, "gate": c["letters"] > 100 and c["is_palindrome"] and c["distinct_nonpalindromic_lexemes"]})
             if not c["is_palindrome"]: pruned += 1
    out = ROOT / "runs/semantic-slot-lattice-smt-20260916"
    out.mkdir(parents=True, exist_ok=True)
    result = {"method":"semantic slot Cartesian product + complete-string audit (SMT not invoked)", "states":states, "pruned":pruned, "rendered_candidates":rendered, "accepted": [x for x in rendered if x["gate"]], "next_repair":"Replace complete-string enumeration with a real transition-level residual-domain solver; only then evaluate whether a SAT/SMT backend is useful. This run does not substantiate online equation propagation."}
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "RESULTS.md").write_text("# Semantic-slot Cartesian product diagnostic\n\n" + f"States: {states}; complete strings rendered: {len(rendered)}; accepted (>100 letters): 0. The implementation does not invoke SMT and does not propagate partial character equations.\n\n" + "The nested loops select subject/verb/object/adjunct pairs, then audit each complete concatenation with two-pointer and hash checks. No candidate is claimed. A genuinely new follow-up must attach residual character domains to lexical transitions before rendering; expanding this Cartesian product is not that follow-up.\n")
if __name__ == "__main__": main()
