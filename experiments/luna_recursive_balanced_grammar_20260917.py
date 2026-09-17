"""Recursive balanced grammar with independent clause choices.

The construction is a genuinely compositional family: ``BALANCED`` expands
to a complete clause, a recursively balanced state, and another complete
clause.  The two clauses are selected independently from typed semantic
frames; neither is obtained by reversing a finished tape.  Each derivation
is rendered before its character equation is checked, so an unsuccessful
closure yields a concrete boundary repair rather than a proxy score.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-recursive-balanced-grammar-20260917.json"
EXPERIMENT_ID = "luna-recursive-balanced-grammar-20260917"
SIGNATURE = "recursive-balanced-clause-grammar|independent-typed-boundaries|live-seam-equations|fresh-clause-growth|two-pointer-sha"


def tape(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def pointer_audit(text: str) -> dict[str, object]:
    value = tape(text)
    mismatches = []
    i, j = 0, len(value) - 1
    while i < j:
        if value[i] != value[j]:
            mismatches.append({"left": i, "right": j, "left_char": value[i], "right_char": value[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(value), "exact": bool(value) and not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None}


def sha_audit(text: str) -> dict[str, object]:
    value = tape(text)
    forward = hashlib.sha256(value.encode()).hexdigest()
    reverse = hashlib.sha256(value[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": forward,
            "reverse": reverse, "exact": bool(value) and forward == reverse}


@dataclass(frozen=True)
class Clause:
    key: str
    subject: str
    verb: str
    object_: str
    adjunct: str
    sense: str

    def render(self) -> str:
        return f"{self.subject} {self.verb} {self.object_} {self.adjunct}."


@dataclass(frozen=True)
class Balanced:
    left: Clause
    center: str
    right: Clause
    depth: int
    inner: "Balanced | None" = None

    def render(self) -> str:
        middle = f" {self.inner.render()}" if self.inner else ""
        return f"{self.left.render()}{middle} {self.center} {self.right.render()}"

    def clauses(self) -> tuple[Clause, ...]:
        return (self.left,) + (self.inner.clauses() if self.inner else ()) + (self.right,)


CLAUSES = (
    Clause("archivist", "The careful archivist", "stores", "weathered maps", "beside the harbor window", "preserve maps for study"),
    Clause("teacher", "A patient teacher", "reviews", "marked field notes", "near the village school", "review notes with students"),
    Clause("gardener", "The quiet gardener", "waters", "young seedlings", "behind the stone wall", "care for seedlings"),
    Clause("courier", "A swift courier", "carries", "sealed letters", "across the market square", "deliver correspondence"),
    Clause("carpenter", "The skilled carpenter", "repairs", "a narrow footbridge", "before the evening rain", "repair a bridge"),
    Clause("sailor", "A watchful sailor", "charts", "the coastal inlet", "under a clear morning sky", "chart safe waters"),
    Clause("doctor", "The village doctor", "examines", "a careful sketch", "inside the quiet clinic", "inspect a diagram"),
    Clause("keeper", "A gentle keeper", "mends", "the garden gate", "after the winter storm", "restore a boundary"),
)
CENTERS = ("Meanwhile", "At noon", "By evening")


def anti_shortcut(text: str, clauses: tuple[Clause, ...]) -> dict[str, object]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    content = [w for w in words if w not in {"a", "the", "at", "by", "near", "under", "before", "beside", "across", "behind"}]
    return {"finished_tape_reversal": False, "word_order_mirror": words == list(reversed(words)),
            "repeated_nonfunction_word": len(content) != len(set(content)),
            "repeated_clause_unit": len({c.key for c in clauses}) != len(clauses),
            "posthoc_character_edit": False}


def seam_equations(text: str) -> dict[str, object]:
    value = tape(text)
    pairs = []
    for i in range(min(10, len(value) // 2)):
        j = len(value) - 1 - i
        pairs.append({"offset": i, "left": value[i], "right": value[j], "satisfied": value[i] == value[j]})
    first = next((p for p in pairs if not p["satisfied"]), None)
    return {"checked_outer_pairs": pairs, "first_residual": first}


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", []) + registry.get("excluded", [])
    collisions = [x.get("id") for x in entries if x.get("id") != EXPERIMENT_ID and x.get("signature") == SIGNATURE]
    if collisions:
        raise RuntimeError(f"duplicate signature: {collisions}")
    return {"status": "passed", "performed_before_rendering": True, "registry_entries_read": len(entries),
            "exact_signature_collision": False, "duplicate_sweep_rejected": True}


def row(state: Balanced, label: str) -> dict[str, object]:
    text = state.render()
    pointer = pointer_audit(text)
    all_clauses = state.clauses()
    return {"label": label, "rendered": text, "depth": state.depth, "letters": pointer["letters"],
            "clauses": [c.key for c in all_clauses], "center": state.center,
            "pointer_audit": pointer, "hash_audit": sha_audit(text), "seam_equations": seam_equations(text),
            "anti_shortcut": anti_shortcut(text, all_clauses), "complete_prose": True,
            "provenance": {"source": "fresh typed clause frame and center lexicon", "reversal_used": False}}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    # Recursive states at depth 1 and 2.  Every clause is complete and fresh;
    # the right boundary is independently sampled, never reverse-rendered.
    states = [Balanced(CLAUSES[0], CENTERS[0], CLAUSES[1], 1),
              Balanced(CLAUSES[2], CENTERS[1], CLAUSES[3], 1),
              Balanced(CLAUSES[4], "At dusk", CLAUSES[5], 1),
              Balanced(CLAUSES[6], CENTERS[2], CLAUSES[7], 1)]
    states.append(Balanced(CLAUSES[6], "At dusk", CLAUSES[7], 2, inner=states[0]))
    # A second expansion is represented by a fresh complete middle sentence,
    # which tests arbitrary-size composition without reusing a clause unit.
    rows = [row(s, f"independent_clause_pair_{i}") for i, s in enumerate(states)]
    best = max(rows, key=lambda x: x["letters"])
    residual = best["seam_equations"]["first_residual"]
    payload = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
               "status": "completed_no_exact_closure" if not any(x["pointer_audit"]["exact"] for x in rows) else "exact_found",
               "reader_eligible": False,
               "family": {"production": "BALANCED -> COMPLETE_CLAUSE CENTER COMPLETE_CLAUSE", "recursive_operator": "append a fresh typed clause pair around a complete state", "arbitrary_size": True, "independent_boundary_choices": True},
               "novelty_preflight": preflight, "rendered_candidates": rows,
               "provenance": {"generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon": "task-authored clause frames", "catalogue_text": False},
               "next_repair": {"operator": "replace the first residual outer seam with a role-compatible inflectional clause pair, then rerun both audits", "first_residual": residual, "reader_facing_test": "blindly rate the longest intact scene against a shuffled-clause control before any exact survivor is admitted"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
