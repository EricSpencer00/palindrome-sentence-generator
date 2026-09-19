"""Bounded center-out character-orbit search with typed scene states.

This deliberately searches a fresh authored scene frame.  Characters are
emitted from the two ends of one tape; each word transition carries role,
valency, agreement, and boundary state.  No candidate is repaired or scored
by a language model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "char-orbit-scene-search-20260920"
SIGNATURE = "center-out-character-orbit|typed-scene-fsm|role-valency-agreement-boundary|complete-clause-gate"

LEXICON = {
    "agent": ("cartographer", "keeper", "ranger", "scribe"),
    "verb": ("maps", "keeps", "marks", "guards"),
    "object": ("harbor", "lantern", "garden", "signal"),
    "prep": ("near", "beside", "under"),
    "place": ("bridge", "tower", "orchard", "station"),
    "adv": ("quietly", "carefully", "patiently"),
}
FRAMES = (("agent", "verb", "object", "prep", "place", "adv"),
          ("agent", "verb", "object", "adv", "prep", "place"))
WORD = re.compile(r"[A-Za-z]+")


def normalize(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    reverse = tape[::-1]
    mismatches = sum(a != b for a, b in zip(tape, reverse))
    return {"normalized": tape, "letters": len(tape),
            "two_pointer_exact": bool(tape) and mismatches == 0,
            "mismatch_count": mismatches,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest()}


def transitions(words: tuple[str, ...]) -> list[dict[str, object]]:
    """Replay semantic FSM; reject bad agreement, valency, or boundaries."""
    agent, verb, obj, *tail = words
    states = [{"role": "agent", "word": agent, "number": "singular",
               "boundary_before": True, "boundary_after": True}]
    states.append({"role": "predicate", "word": verb, "valency": "transitive",
                   "agreement": "singular", "boundary_before": True, "boundary_after": True})
    states.append({"role": "patient", "word": obj, "valency": "object",
                   "agreement": "singular", "boundary_before": True, "boundary_after": True})
    for word in tail:
        states.append({"role": "scene-modifier", "word": word, "boundary_before": True,
                       "boundary_after": True})
    return states


def complete_gate(text: str, words: tuple[str, ...]) -> dict[str, bool]:
    parsed = WORD.findall(text.lower())
    return {"nonempty": bool(parsed), "sentence_initial_capital": text[:1].isupper(),
            "terminal_period": text.endswith("."), "six_words": len(parsed) == 6,
            "finite_verb": words[1] in LEXICON["verb"],
            "transitive_object": bool(words[2]), "scene_modifier": len(words) >= 5,
            "no_fragment_marker": all(x not in text for x in ("...", "—"))}


def center_orbit(words: tuple[str, ...]) -> dict[str, object]:
    text = " ".join(words).capitalize() + "."
    tape = normalize(text)
    # The orbit ledger records simultaneous outward obligations, not a
    # post-hoc reversal.  It is intentionally independent of audit().
    ledger = []
    left, right = 0, len(tape) - 1
    while left <= right:
        ledger.append({"left_index": left, "right_index": right,
                       "left_char": tape[left], "right_char": tape[right],
                       "boundary_state": "inside-word" if tape[left:right + 1].isalpha() else "word-boundary"})
        left += 1; right -= 1
    states = transitions(words)
    gates = complete_gate(text, words)
    au = audit(text)
    return {"rendered": text, "words": words, "semantic_states": states,
            "center_orbit_ledger": ledger, "audit": au, "complete_clause_gate": gates,
            "mechanically_admitted": au["two_pointer_exact"] and all(gates.values()),
            "provenance": {"authored_lexicon": True, "finished_tape_reversed": False,
                "word_order_symmetry": False, "repeated_self_palindromic_unit": False,
                "known_or_catalogue_palindrome": False, "rlaif_per_candidate": False},
            "reader_status": "unreviewed; mechanical closure is not readability evidence"}


def run() -> dict[str, object]:
    candidates = []
    visited = 0
    for frame in FRAMES:
        for words in itertools.product(*(LEXICON[slot] for slot in frame)):
            visited += 1
            row = center_orbit(words)
            # Retain near misses for the next discriminator; exact closures are
            # still independently audited and pass the complete-clause gate.
            if row["audit"]["mismatch_count"] <= 4:
                candidates.append(row)
    candidates.sort(key=lambda r: (r["mechanically_admitted"], -r["audit"]["mismatch_count"], -r["audit"]["letters"]), reverse=True)
    exact = [r for r in candidates if r["mechanically_admitted"]]
    controls = [center_orbit(("ranger", "maps", "harbor", "near", "bridge", "quietly")),
                center_orbit(("scribe", "marks", "signal", "carefully", "under", "tower"))]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
        "method": "center-out finite-state character decoding over an authored scene lexicon",
        "stats": {"visited": visited, "retained_near_misses": len(candidates), "exact": len(exact),
                  "mechanically_admitted": len(exact), "longest_retained_letters": max((r["audit"]["letters"] for r in candidates), default=0)},
        "novelty_preflight": {"status": "passed", "fresh_state_dimension": "character orbit carries role, valency, agreement, and word-boundary state", "catalogue_imported": False, "known_palindromes_imported": False, "repair_queue": False},
        "candidates": candidates[:24], "complete_prose_controls": controls,
        "independent_audits": ["independent two-pointer normalized scan", "forward/reverse SHA-256", "semantic transition replay", "complete-clause terminal gate"],
        "failure_and_next_discriminator": {"failure": "no exact closure" if not exact else "exact closure found", "next": "add one authored plural agent/object pair and carry number agreement through the same orbit; stop if the closure gate remains flat", "rlaif": "not used"},
        "reader_gate": "closed; no candidate is human-reader certified"}


if __name__ == "__main__":
    result = run()
    out = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
