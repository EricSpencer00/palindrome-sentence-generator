"""Rejected fixed-frame diagnostic over the live word-residual product.

This file originally described its depth loop as recursive discourse growth.
That was wrong: every iteration starts a fresh residual search against the
same right frame, so no emitted clause or character debt crosses an iteration.
The retained run is useful only as evidence that four independent frame probes
do not implement recursive paragraph composition.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from llm_palindrome.dual_parse import word_residual_search, letter_tape
from llm_palindrome.admission import mechanical_admission_checks

ID = "recursive-discourse-grammar-20260921"

# Each transition is a discourse move, not a mirrored lexical unit.
TRANSITIONS = {
    "observe": ("explain", "contrast"),
    "explain": ("contrast", "conclude"),
    "contrast": ("observe", "conclude"),
    "conclude": ("observe",),
}
FRAMES = {
    "observe": (("A:determiner", ("an",)), ("A:agent", ("aide",)),
                ("A:verb", ("rips",)), ("A:quantity", ("nine",)),
                ("A:object", ("memos",))),
    "explain": (("A:determiner", ("a",)), ("A:agent", ("pilot",)),
                 ("A:verb", ("maps",)), ("A:quantity", ("two",)),
                 ("A:object", ("caves",))),
    "contrast": (("A:determiner", ("the",)), ("A:agent", ("scribe",)),
                  ("A:verb", ("marks",)), ("A:quantity", ("five",)),
                  ("A:object", ("glyphs",))),
    "conclude": (("A:determiner", ("a",)), ("A:agent", ("reader",)),
                  ("A:verb", ("weighs",)), ("A:quantity", ("six",)),
                  ("A:object", ("claims",))),
}
RIGHT = (("B-prime:response", ("some",)), ("B-prime:agent", ("men",)),
         ("B-prime:verb", ("inspire",)), ("A-prime:patient", ("Diana",)))

def _audit(text: str) -> dict:
    tape = letter_tape(text)
    return {"letters": len(tape), "exact": tape == tape[::-1],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run(max_depth: int = 4, max_states: int = 4000) -> dict:
    state = {"phase": "observe", "topic": "evidence", "tense": "present", "polarity": "affirmative"}
    frontiers, outputs, seen_content = [], [], set()
    for depth in range(1, max_depth + 1):
        phase = state["phase"]
        left = FRAMES[phase]
        search = word_residual_search(left, RIGHT, max_states=max_states,
                                      max_results=20, reject_intermediate_closure=True)
        for dead in search["dead_frontiers"][:8]:
            frontiers.append({"depth": depth, "semantic_state": dict(state),
                              "dead_frontier": dead,
                              "reject_intermediate_closure": True})
        for result in search["results"]:
            text = result["rendered"]
            words = set(text.casefold().split())
            unique = not (words & seen_content)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
            row = {"depth": depth, "semantic_state": dict(state), "rendered": text,
                   "audit": _audit(text), "mechanical_admission": checks,
                   "unique_content": unique,
                   "central_admission": all(checks.values()),
                   "provenance": {"recursive_clause_transition": False,
                                  "loop_depth": depth, "semantic_state_carried": False,
                                  "composed_across_depths": False,
                                  "independent_frame_query": True,
                                  "word_residual_product": True,
                                  "reject_intermediate_closure": True,
                                  "finished_tape_reversal": False,
                                  "completed_prose_enumeration": False,
                                  "exact_subpalindrome_composition": False}}
            if unique and row["central_admission"]:
                outputs.append(row); seen_content.update(words)
        state = {**state, "phase": TRANSITIONS[phase][(depth - 1) % len(TRANSITIONS[phase])]}
    return {"experiment_id": ID,
            "method": "independent typed-frame diagnostics over online word residuals",
            "target_depth": max_depth,
            "outputs": outputs, "frontiers": frontiers,
            "stats": {"depths": max_depth, "frontiers": len(frontiers), "outputs": len(outputs)},
            "provenance": {"recursive_loop": False, "central_admission": True,
                           "reject_intermediate_closure": True, "unique_content": True,
                           "semantic_clause_state": False,
                           "composed_across_depths": False,
                           "independent_frame_queries": True,
                           "finished_tape_reversal": False,
                           "completed_prose_enumeration": False,
                           "exact_subpalindrome_composition": False},
            "status": "rejected_as_independent_frame_loop",
            "next_operator": "Carry grammar phase and nonempty character debt in one recurrent product state; detect a coaccessible productive cycle instead of restarting each frame."}

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "runs" / f"{ID}.json"
    out.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))
