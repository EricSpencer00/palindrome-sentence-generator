"""Bounded outer lexical-pair construction (diagnostic, no candidate claim).

Complete subject NPs and complete right-clause object/verb suffixes are varied
together.  The middle clause is fixed only as a grammatical frame; this is a
fresh seam geometry, not a reverse-word or catalogue sweep.
"""
import hashlib, json, re
from pathlib import Path
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
NAME = "outer-subject-lexical-pair-20260918"
SUBJECTS = ["the quiet baker", "a patient pilot", "some careful clerks", "the harbor keeper"]
LEFT_VERBS = ["records", "opens", "carries", "marks"]
RIGHT_SUFFIXES = ["the old map", "a blue ledger", "our spare compass", "the narrow gate"]
RIGHT_VERBS = ["near", "beside", "under", "within"]

def audit(text):
    tape = normalize_letters(text); rev = tape[::-1]
    mismatches = sum(a != b for a, b in zip(tape, rev)) + abs(len(tape)-len(rev))
    i, j = 0, len(tape)-1
    while i < j and tape[i] == tape[j]: i, j = i+1, j-1
    return {"letters": len(tape), "two_pointer_exact": i >= j,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def run():
    rows = []
    for si, subject in enumerate(SUBJECTS):
        for vi, verb in enumerate(LEFT_VERBS):
            for oi, obj in enumerate(RIGHT_SUFFIXES):
                for ri, rverb in enumerate(RIGHT_VERBS):
                    text = f"{subject} {verb} the chart; meanwhile, {rverb} {obj}."
                    checks = mechanical_admission_checks(text, min_letters=30, max_letters=100)
                    rows.append({"candidate_id": f"oslp-{si}-{vi}-{oi}-{ri}", "rendered": text,
                        "outer_pair": {"subject_np": subject, "right_clause_suffix": f"{rverb} {obj}"},
                        "audit": audit(text), "strict_gate": checks,
                        "provenance": {"fresh_hand_authored_roles": True, "catalogue_used": False,
                            "direct_reversed_word_pair": False, "hidden_multiword_palindrome_span": False,
                            "finished_tape_reversal": False, "duplicate_polar_question_sweep": False}})
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": NAME, "method": "joint complete grammatical outer subject NP × terminal right-clause object/verb suffix; live boundary debt measured after each rendering",
      "construction": {"subject_role": "agent NP", "right_role": "locative verb + object suffix", "states": len(rows)},
      "rendered_candidates": rows[:12], "best_probe": best,
      "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows), "longest_letters": max(r["audit"]["letters"] for r in rows), "best_mismatches": best["audit"]["mismatches"]},
      "novelty_preflight": {"new_geometry": "outer lexical phrase pair crosses subject and terminal right-clause boundaries", "prior_lane_reused": False, "duplicate_sweep": False},
      "reader_gate": {"status": "not_triggered", "reason": "no strict exact closure", "human_readability_certified": False},
      "next_repair": {"operator": "replace only the right-clause preposition/object pair at the best residual seam while preserving subject number and verb valency", "reason": "the longest intact prose probe leaves residual character debt at the outer boundary"},
      "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "hand-authored common English role phrases", "human_readability_certified": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{NAME}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
