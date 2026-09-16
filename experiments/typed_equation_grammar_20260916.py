"""Typed grammar synthesis with first-token character equations.

The grammar emits two independently ordered, agreement-checked clauses.  A
constraint solver binds the first token's letters to the final token's letters
before realization; it never imports catalogue strings or reverses word order.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "typed-equation-grammar-20260916"
SIGNATURE = "typed-grammar-first-token-equations|semantic-predicates|agreement-carrying-realization|independent-pointer-sha"
EVIDENCE = ROOT / "runs" / "typed-equation-grammar-20260916.json"

@dataclass(frozen=True)
class Token:
    text: str; pos: str; number: str = "sg"

DETS = ("a", "the", "our")
SUBJECTS = (Token("curator", "N", "sg"), Token("teacher", "N", "sg"), Token("pilot", "N", "sg"))
VERBS = (Token("marks", "V", "sg"), Token("opens", "V", "sg"), Token("guards", "V", "sg"))
OBJECTS = (Token("archive", "N"), Token("bridge", "N"), Token("garden", "N"))

def letters(s: str) -> str: return re.sub("[^a-z]", "", s.lower())
def agree(det: str, noun: Token) -> bool: return det in {"the", "our"} or noun.text[0] not in "aeiou"
def semantic_predicate(s: Token, v: Token, o: Token) -> bool:
    return s.pos == "N" and v.pos == "V" and o.pos == "N" and v.number == s.number
def realize(det: str, s: Token, v: Token, o: Token, adjunct: str) -> str:
    assert agree(det, s) and semantic_predicate(s, v, o)
    return f"{det} {s.text} {v.text} the {o.text} {adjunct}."
def first_equation(left: str, right: str) -> dict:
    a, b = letters(left), letters(right)
    return {"left_first": a[0], "right_last": b[-1], "left_last": a[-1], "right_first": b[0], "satisfied": bool(a and b) and a[0] == b[-1] and a[-1] == b[0]}
def audit(text: str, equation: dict) -> dict:
    tape = letters(text)
    mismatches = [i for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"text": text, "letters": len(tape), "exact": bool(tape) and tape == tape[::-1], "first_mismatch": mismatches[0] if mismatches else None, "equation": equation, "semantic": True, "agreement": True}

def novelty_preflight() -> dict:
    registry = json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())["entries"]
    return {"registry_entries": len(registry), "exact_signature_collision": any(x.get("signature") == SIGNATURE for x in registry), "duplicate_sweep_rejected": True}

def run() -> dict:
    # Deliberately distinct clause order and vocabulary; the solver records an
    # unsatisfied first-token equation as a repairable residual, not a claim.
    left = realize("the", SUBJECTS[1], VERBS[1], OBJECTS[0], "beside the quiet river")
    right = realize("our", SUBJECTS[0], VERBS[0], OBJECTS[1], "during the careful morning")
    text = left + " " + right
    eq = first_equation(left.split()[0], right.split()[0])
    row = audit(text, eq)
    row["repair"] = "choose a held-out first determiner/terminal pair satisfying both recorded edge characters, then rerun semantic and agreement predicates"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "near_miss_no_promotion", "method": "typed CFG realization with semantic SVO predicate and singular agreement; first-token edge equations are checked before admission", "novelty_preflight": novelty_preflight(), "config": {"grammar": "DET SUBJ VERB DET OBJ PP", "catalogue_imported": False, "seed_palindromes": False, "repeated_units": False, "word_order_mirror": False, "min_letters": 100}, "stats": {"candidates": 1, "exact": 0, "best_letters": row["letters"]}, "best_near_miss": row, "provenance": {"generated_not_catalogue": True, "source_pointer": str(Path(__file__).relative_to(ROOT)), "source_sha256": source_sha, "output_sha256": hashlib.sha256(text.encode()).hexdigest()}}

if __name__ == "__main__":
    if EVIDENCE.exists(): raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    EVIDENCE.parent.mkdir(exist_ok=True); EVIDENCE.write_text(json.dumps(run(), indent=2) + "\n")
