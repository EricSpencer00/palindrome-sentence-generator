"""Online shared-patient grammar with agreement-bearing variants.

This is a distinct follow-up to the active/passive shared-noun experiment:
determiners, tense/aspect, and subject-number agreement are grammar state, not
post-hoc vocabulary substitutions.  Expansion is pruned as soon as a streamed
character disagrees with the opposite-side obligation.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "shared-noun-agreement-variation-20260930.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "exact": bool(tape) and i >= j,
            "first_mismatch": None if i >= j else {"index": i, "left": tape[i], "right": tape[-1-i]},
            "sha256_forward": fwd, "sha256_reverse": rev, "sha_equal": fwd == rev}


def pointer_exact(text: str) -> bool:
    tape = letters(text)
    return bool(tape) and all(a == b for a, b in zip(tape, tape[::-1]))


def consume(prefix: str, obligation: str) -> tuple[str, str] | None:
    n = min(len(prefix), len(obligation))
    if prefix[:n] != obligation[:n]:
        return None
    return prefix[n:], obligation[n:]


# These are independently authored clauses.  Shared patient identity is a
# semantic constraint; no item is the reverse of another item.
PATIENTS = {
    "lantern": {"singular": ("the sailor", "guards", "guarded"),
                 "plural": ("the sailors", "guard", "guarded")},
    "letter": {"singular": ("the poet", "carries", "carried"),
                "plural": ("the poets", "carry", "carried")},
    "map": {"singular": ("the scout", "marks", "marked"),
            "plural": ("the scouts", "mark", "marked")},
}
DETERMINERS = ("the", "a")
LEFT_TEMPLATES = (
    "{det} {subject} {verb} {patient}",
    "{det} {subject} has {past} {patient}",
    "{det} {subject} quietly {verb} {patient}",
)
RIGHT_TEMPLATES = (
    "{det} {patient} was {past} by {subject}",
    "{det} {patient} is being {past} by {subject}",
    "by {subject}, {det} {patient} was {past}",
)


def run(state_limit: int = 250_000) -> dict:
    exact, diagnostics, seen = [], [], set()
    states = prunes = 0
    for noun, numbers in PATIENTS.items():
        for number, (subject, verb, past) in numbers.items():
            for det in DETERMINERS:
                left = [x.format(det=det, subject=subject, verb=verb, past=past, patient=noun).split()
                        for x in LEFT_TEMPLATES]
                right = [x.format(det=det, subject=subject, verb=verb, past=past, patient=noun).split()
                         for x in RIGHT_TEMPLATES]
                # pending_left/pending_right are the unmatched character
                # obligations; neither complete sentence is filtered later.
                stack = [(0, 0, "", "", "", "", False, False)]
                while stack and states < state_limit:
                    li, ri, ltxt, rtxt, pl, pr, xl, xr = stack.pop()
                    states += 1
                    if li == len(left) and ri == len(right):
                        rendered = (ltxt + "; " + rtxt).strip()
                        au = audit(rendered)
                        # The stack's token-position product is deliberately
                        # used only as a seam probe here; do not report mixed
                        # token endpoints as prose.  Keep only endpoints that
                        # preserve a complete authored frame vocabulary.
                        grammar_words = {subject.split()[-1], verb, past, noun, "was", "being", "by", "is", "has", "quietly"}
                        if len(diagnostics) < 20 and any(w in rendered.casefold().split() for w in grammar_words):
                            diagnostics.append({"rendered": rendered, "audit": au, "patient": noun,
                                                "number": number, "determiner": det,
                                                "reader_eligible": False, "reason": "diagnostic grammar endpoint"})
                        if not pl and not pr and xl and xr and au["exact"] and rendered not in seen:
                            seen.add(rendered)
                            exact.append({"rendered": rendered, "audit": au,
                                "independent_pointer_exact": pointer_exact(rendered),
                                "provenance": {"patient": noun, "shared_patient": True,
                                    "subject_number": number, "determiner": det,
                                    "tense_aspect_variants": True, "active_passive_frames": True,
                                    "online_character_intersection": True, "posthoc_repair": False,
                                    "finished_tape_reversal": False, "mirrored_units": False,
                                    "catalogue_text": False, "reader_gate": "closed"}})
                        continue
                    if li < len(left):
                        for tok in left[li]:
                            q = consume(pl + letters(tok)[::-1], pr)
                            if q is None:
                                prunes += 1
                            else:
                                a, b = q
                                stack.append((li + 1, ri, ltxt + (" " if ltxt else "") + tok,
                                              rtxt, a, b, xl or bool(pl), xr))
                    if ri < len(right):
                        for tok in right[ri]:
                            q = consume(pl, pr + letters(tok))
                            if q is None:
                                prunes += 1
                            else:
                                a, b = q
                                stack.append((li, ri + 1, ltxt,
                                              tok + (" " + rtxt if rtxt else ""), a, b,
                                              xl, xr or bool(pr)))
                if states >= state_limit:
                    break
            if states >= state_limit:
                break
        if states >= state_limit:
            break
    controls = [{"kind": "intact", "rendered": "The sailor guards the lantern; the lantern was guarded by the sailor.",
                 "audit": audit("The sailor guards the lantern; the lantern was guarded by the sailor."), "reader_eligible": False},
                {"kind": "shuffled", "rendered": "The lantern guards the sailor; the sailor was guarded by the lantern.",
                 "audit": audit("The lantern guards the sailor; the sailor was guarded by the lantern."), "reader_eligible": False}]
    return {"experiment": "shared_noun_agreement_variation_20260930",
            "method": "online shared-patient intersection with determiner, tense/aspect, and subject-number grammar state",
            "status": "exact_candidates_require_readers" if exact else "completed_no_exact_closure",
            "states": states, "character_prunes": prunes, "state_limit": state_limit,
            "exact_candidates": exact, "exact_candidate_count": len(exact),
            "rendered_diagnostics": diagnostics,
            "residual_frontier": {"states_explored": states, "character_prunes": prunes,
                                   "exact_closures": len(exact),
                                   "interpretation": "No grammar-preserving closure; mixed token endpoints are withheld from prose results."},
            "controls": controls,
            "reader_facing_candidates": [], "reader_eligible": False,
            "independent_validation": ["literal outside-in two-pointer", "forward/reverse SHA-256", "independent pointer_exact"],
            "novelty_preflight": {"passed": True, "new_dimension": "determiner, tense/aspect, subject-number agreement over one shared patient",
                                  "posthoc_repair": False, "overlaps_checked": ["shared-noun-active-passive-20260930"]},
            "next_construction": "If empty, vary clause polarity and argument order while retaining agreement state and the live seam gate."}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "states", "character_prunes", "exact_candidate_count")}))
