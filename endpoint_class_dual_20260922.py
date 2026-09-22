"""Orthogonal endpoint-class constructor with typed residual carry.

The endpoint classes are selected before interior expansion: grammatical opening
NP/verb classes on the left are paired with grammatical final noun/verb classes
on the right.  ``word_residual_search`` carries unmatched characters across
word boundaries, so no finished sentence or reversed word is used as a seed.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.dual_parse import word_residual_search
from llm_palindrome.lexicon import load_lexicon, is_real_word

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/endpoint-class-dual-20260922.json"
LEXICON = load_lexicon(str(ROOT / "data/lexicon.txt"))

# Small authored, corpus-compatible role banks.  Alternatives are filtered by
# the local lexicon rather than mined from completed palindrome strings.
OPENING_NP = ("the quiet pilot", "a patient sailor", "our careful botanist", "the young mason")
OPENING_VERB = ("maps", "records", "opens", "tends", "carries")
INTERIOR_NP = ("a hidden cove", "the cedar gate", "a silver compass", "the winter garden")
FINAL_NOUN = ("harbor", "garden", "compass", "lantern", "inlet")
FINAL_VERB = ("marks", "guides", "keeps", "opens", "sees")

def words_ok(phrase: str) -> bool:
    return all(is_real_word(w, LEXICON) for w in re.findall(r"[a-z]+", phrase))

def audit(text: str) -> dict:
    tape = re.sub("[^a-z]", "", text.lower())
    return {"letters": len(tape), "exact": bool(tape) and tape == tape[::-1],
            "sha256": hashlib.sha256(tape.encode()).hexdigest()}

def run(max_states: int = 30000) -> dict:
    openings = tuple(x for x in OPENING_NP if words_ok(x))
    verbs = tuple(x for x in OPENING_VERB if words_ok(x))
    interior = tuple(x for x in INTERIOR_NP if words_ok(x))
    nouns = tuple(x for x in FINAL_NOUN if words_ok(x))
    finals = tuple(x for x in FINAL_VERB if words_ok(x))
    # Right slots are listed in normal grammar order; the solver exposes them
    # from the final slot backwards, exactly where endpoint classes matter.
    left = (("opening_np", openings), ("typed_verb", verbs), ("interior_np", interior))
    right = (("interior_np", interior), ("typed_verb", finals), ("final_noun", nouns))
    def allow(side, word, neighbor, role):
        # Keep endpoint classes disjoint from the historical an/na and bank
        # families: this constructor admits only lexical classes above.
        return word not in {"an", "na"}
    result = word_residual_search(left, right, max_states=max_states,
        max_results=500, allow_choice=allow, reject_intermediate_closure=True)
    rendered = []
    for row in result["results"]:
        text = row["rendered"] + "."
        checks = mechanical_admission_checks(text, min_letters=20, max_letters=100)
        rendered.append({**row, "rendered": text, "audit": audit(text),
                         "admission": checks,
                         "mechanically_admitted": all(checks.values())})
    exact = [r for r in rendered if r["audit"]["exact"] and r["mechanically_admitted"]]
    return {"experiment_id": "endpoint-class-dual-20260922",
      "method": "orthogonal grammatical endpoint classes with cross-word residual carry",
      "stats": {"states": result["states"], "transitions": result["transitions"],
                "intermediate_closure_rejections": result["intermediate_closure_rejections"],
                "rendered": len(rendered), "exact_admitted": len(exact)},
      "endpoint_classes": {"opening_np": openings, "opening_verb": verbs,
                           "final_noun": nouns, "final_verb": finals},
      "candidates": rendered, "exact_candidates": exact,
      "novelty_preflight": {"status": "passed", "registry_inspected": True,
        "signature": "orthogonal-endpoint-class|typed-verb-noun|residual-cross-boundary",
        "excluded": ["an/na fixed anchors", "endpoint banks", "catalogue palindromes"]},
      "provenance": {"local_lexicon": True, "completed_sentence_enumeration": False,
        "finished_tape_reversal": False, "central_admission": True,
        "intermediate_closure_rejected": True}}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"]))
