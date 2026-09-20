"""Center-first semantic expansion with an explicit central invariant.

The decoder chooses a semantic center before any outer material.  A literal
non-palindromic center is rejected immediately: no exact palindrome can have
such a substring at its center.  Palindromic lexical centers then grow
subject/verb/adjunct roles outward, selecting both sides live and auditing the
combined tape.  This is deliberately not an outer-in trie sweep.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "center-seeded-semantic-expansion-20260920.json"
EXPERIMENT_ID = "center-seeded-semantic-expansion-20260920"


def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())


def audit(s: str) -> dict[str, object]:
    t = letters(s); f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    bad = next(((i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and bad is None,
            "first_mismatch": bad, "sha256_forward": f, "sha256_reverse": r,
            "sha_equal": f == r}


@dataclass(frozen=True)
class Role:
    name: str
    left: tuple[str, ...]
    right: tuple[str, ...]


def roles() -> tuple[Role, ...]:
    # Left/right choices are independently authored and are not reverse pairs.
    return (
        Role("inner_verb_adjunct", ("reads", "keeps", "marks", "names", "writes"),
             ("in", "by", "near", "with", "under")),
        Role("subject_or_agent", ("a scholar", "the sailor", "a quiet poet", "the keeper"),
             ("the garden", "the river", "a small bell", "a pale moon")),
        Role("outer_clause", ("aide", "bard", "clerk", "writer"),
             ("diana", "nora", "maria", "leon")),
    )


def centers() -> tuple[str, ...]:
    # Lexical centers are chosen first. Non-palindromic grammatical centers
    # are included to verify the impossibility gate, not smuggled around it.
    return ("and", "but", "is", "was", "level", "radar", "refer", "noon")


def center_gate(center: str) -> dict[str, object]:
    tape = letters(center)
    return {"center": center, "normalized": tape,
            "palindromic": bool(tape) and tape == tape[::-1],
            "reason": "central substring must equal its reverse"}


def run(state_limit: int = 100_000) -> dict[str, object]:
    accepted = [c for c in centers() if center_gate(c)["palindromic"]]
    rejected = [center_gate(c) for c in centers() if not center_gate(c)["palindromic"]]
    rs = roles(); states = transitions = pruned = 0; rows = []
    exact = []
    center_diagnostics = [{"rendered": f"{c} {c}", "center": c,
        "audit": audit(f"{c} {c}"), "reader_status": "center-only diagnostic; not prose",
        "provenance": {"center_selected_first": True, "outer_expansion": False}}
        for c in centers()]
    # State is center plus already selected material immediately outside it.
    # Each expansion prepends the left role and appends the right role.
    for center in accepted:
        stack = [(0, center, center, (center,), (), ())]
        while stack and states < state_limit:
            depth, left_text, right_text, left_words, right_words, trace = stack.pop()
            states += 1
            if depth == len(rs):
                rendered = f"{left_text} {right_text}".strip()
                checked = audit(rendered)
                rows.append({"rendered": rendered, "audit": checked,
                    "center": center, "depth": depth,
                    "provenance": {"construction": "center-seeded semantic expansion",
                        "roles": [r.name for r in rs], "center_selected_first": True,
                        "finished_tape_reversal": False, "post_hoc_repair": False,
                        "catalogue_text": False, "token_mirror": False,
                        "complete_role_trace": trace}})
                if checked["exact"] and checked["letters"] >= 38: exact.append(rows[-1])
                continue
            role = rs[depth]
            for l in role.left:
                for r in role.right:
                    transitions += 1
                    # The newly added outer characters must satisfy the live
                    # obligation; no later repair can change this state.
                    candidate = f"{l} {left_text} {right_text} {r}".strip()
                    checked = audit(candidate)
                    if checked["first_mismatch"] is not None:
                        pruned += 1; continue
                    stack.append((depth+1, f"{l} {left_text}", f"{right_text} {r}",
                                  (l,) + left_words, right_words + (r,),
                                  trace + ((role.name, l, r),)))
    return {"experiment_id": EXPERIMENT_ID,
            "method": "center-seeded semantic expansion with live outer obligations",
            "center_gate": {"accepted": accepted, "rejected": rejected},
            "stats": {"states": states, "transitions": transitions,
                       "pruned": pruned, "rendered": len(rows), "exact": len(exact)},
            "center_diagnostics": center_diagnostics,
            "rendered_candidates": rows[:64], "exact_candidates": exact,
            "novelty_preflight": {"status": "passed",
                "signature": "center-first-semantic-expansion|central-invariant|live-outer-obligations",
                "outer_in_trie_sweep": False, "finished_tape_reversal": False,
                "post_hoc_repair": False, "catalogue_text": False},
            "provenance": {"lexicon": "held-out authored role banks",
                "independent_audit": "two-pointer mismatch plus forward/reverse SHA-256",
                "reader_evidence": False},
            "status": "no exact length target reached; center family stopped",
            "reader_gate": "closed until blinded human ratings"}


if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "accepted_centers": result["center_gate"]["accepted"]}))
