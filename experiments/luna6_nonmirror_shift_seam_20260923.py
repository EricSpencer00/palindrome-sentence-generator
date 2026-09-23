#!/usr/bin/env python3
"""Small exact feasibility probe for two non-mirrored insertions in the 568 tape.

The operator inserts equal-length strings at two independently selected word
boundaries.  Unlike a reflected-cut pair, the interval between the cuts sees a
shifted reflection map.  This script asks whether any such topology is even
character-feasible before attempting English lexicalization.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs/luna6-nonmirror-shift-seam-20260923.json"
EXPECTED_PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def outside_in(text: str) -> dict:
    tape = letters(text)
    mismatch = next((i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    return {"letters": len(tape), "exact": mismatch is None, "first_mismatch": mismatch}


def project_validation(text: str) -> dict:
    # Independent project check imported lazily from the repository's validator.
    from llm_palindrome.validator import is_palindrome

    return {"exact": bool(is_palindrome(text))}


def word_boundaries(text: str) -> list[int]:
    tape_pos = 0
    cuts = [0]
    for match in re.finditer(r"[A-Za-z]+|[^A-Za-z]+", text):
        token = match.group()
        if token.isalpha():
            tape_pos += len(token)
            cuts.append(tape_pos)
    return sorted(set(cuts))


def source_offset_for_tape_cut(text: str, cut: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isalpha():
            seen += 1
            if seen == cut:
                return index + 1
    if cut == seen:
        return len(text)
    raise ValueError(f"cut {cut} is not a source boundary")


def feasible_fixed_constraints(parent: str, left_cut: int, right_cut: int, width: int) -> dict:
    """Treat inserted symbols as variables; reject fixed/fixed contradictions."""
    # A sentinel is one variable symbol at each inserted position.
    candidate = parent[:left_cut] + "\0" * width + parent[left_cut:right_cut] + "\0" * width + parent[right_cut:]
    n = len(candidate)
    fixed_conflicts = []
    variable_pairs = 0
    unresolved = []
    for i in range(n // 2):
        j = n - 1 - i
        a, b = candidate[i], candidate[j]
        if a == "\0" or b == "\0":
            variable_pairs += 1
            if a == "\0" and b == "\0":
                unresolved.append([i, j])
        elif a != b:
            fixed_conflicts.append({"left_offset": i, "right_offset": j, "required": [a, b]})
            break
    return {
        "feasible": not fixed_conflicts,
        "first_fixed_conflict": fixed_conflicts[0] if fixed_conflicts else None,
        "variable_pairs": variable_pairs,
        "variable_variable_pairs": len(unresolved),
        "first_cursor": (fixed_conflicts[0]["left_offset"] if fixed_conflicts else None),
    }


def main() -> None:
    parent_doc = json.loads(PARENT_PATH.read_text())
    parent_text = parent_doc["rows"][0]["rendered"]
    parent_tape = letters(parent_text)
    if sha(parent_tape) != EXPECTED_PARENT_SHA:
        raise SystemExit("pinned parent SHA mismatch")
    if not outside_in(parent_text)["exact"] or not project_validation(parent_text)["exact"]:
        raise SystemExit("pinned parent failed independent exactness checks")

    # Restrict to word boundaries in the outer half and a modest width range.
    # Exclude the equal reflected-cut topology and ensure a real intervening
    # retained interval exists.  This is a feasibility diagnostic, not a phrase
    # sweep or an English-readability search.
    cuts = [c for c in word_boundaries(parent_text) if 12 <= c <= len(parent_tape) - 12]
    results = []
    best = None
    for left in cuts:
        for right in cuts:
            if right <= left or right - left < 12:
                continue
            if right == len(parent_tape) - left:
                continue
            for width in range(1, 13):
                check = feasible_fixed_constraints(parent_tape, left, right, width)
                item = {"left_cut": left, "right_cut": right, "insert_width_each": width, **check}
                if best is None or (check["first_cursor"] or len(parent_tape)) > (best["first_cursor"] or len(parent_tape)):
                    best = item
                if check["feasible"]:
                    results.append(item)

    # Render one representative feasible topology using the only strings its
    # character equations permit.  This deliberately exposes whether a
    # character-feasible state actually lexicalizes as prose.
    representative = results[0] if results else None
    forced_render = None
    if representative:
        left, right, width = (representative[k] for k in ("left_cut", "right_cut", "insert_width_each"))
        candidate_tape = parent_tape[:left] + "\0" * width + parent_tape[left:right] + "\0" * width + parent_tape[right:]
        assigned = [None] * len(candidate_tape)
        for pos in range(len(candidate_tape) // 2):
            opp = len(candidate_tape) - 1 - pos
            a, b = candidate_tape[pos], candidate_tape[opp]
            if a == "\0":
                assigned[pos] = b
            if b == "\0":
                assigned[opp] = a
        left_text = "".join(assigned[left:left + width])
        right_start = right + width
        right_text = "".join(assigned[right_start:right_start + width])
        left_src = source_offset_for_tape_cut(parent_text, left)
        right_src = source_offset_for_tape_cut(parent_text, right)
        forced_render = parent_text[:right_src] + " " + right_text + parent_text[right_src:]
        forced_render = forced_render[:left_src] + " " + left_text + forced_render[left_src:]
        rendered_tape = letters(forced_render)
        left_words = ["nora", "nadia"] if left_text == "noranadia" else None
        right_words = ["aidan", "aron"] if right_text == "aidanaron" else None
        forced_render = {
            "left_insert": left_text,
            "right_insert": right_text,
            "rendered": forced_render,
            "normalized_letters": len(rendered_tape),
            "normalized_sha256": sha(rendered_tape),
            "outside_in": outside_in(forced_render),
            "project_validator": project_validation(forced_render),
            "boundary_audit": {
                "left_partition": left_words,
                "right_partition": right_words,
                "whole_token_reversal_pairs": [["nora", "aron"], ["nadia", "aidan"]] if left_words and right_words else "not manually classified",
                "shortcut_free": False if left_words and right_words else "not established",
            },
            "local_context": [parent_text[max(0, left_src - 28):left_src], parent_text[max(0, right_src - 28):right_src]],
            "readability_status": "rejected diagnostic only: forced insertions are proper-name fragments adjacent to duplicate context; not coherent English clauses",
        }

    # For any candidate topology the full insertion text would have to be
    # independently authored English and survive the reflected-boundary audit.
    # Do not lexicalize a topology whose only evidence is a wildcard equation.
    record = {
        "experiment_id": "luna6-nonmirror-shift-seam-20260923",
        "status": "no_lexicalized_child",
        "hypothesis": "two insertions at non-reflected word boundaries can carry a shifted character debt through retained parent material",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(parent_tape),
            "normalized_sha256": sha(parent_tape),
            "rendered": parent_text,
            "outside_in": outside_in(parent_text),
            "project_validator": project_validation(parent_text),
        },
        "novelty_preflight": {
            "rejected_reflected_identity_pair": "runs/audit-incumbent-568-identity-witness-growth-20260923.json",
            "separate_boundary_event_resegmentation": "runs/incumbent-568-boundary-event-residual-20260923.json",
            "operator_difference": "two cuts are independently selected, with retained material between them tested under the shifted reflection map; no clause-pair mirroring, event-chain expansion, or reflected-cut substitution",
        },
        "search": {
            "cut_domain": "normalized word boundaries in [12,556]",
            "cut_count": len(cuts),
            "insert_width_each": [1, 12],
            "tested_topologies": sum(1 for left in cuts for right in cuts if right > left and right - left >= 12 and right != len(parent_tape) - left) * 12,
            "character_feasible_nonreflected_topologies": results,
            "best_obstruction": best,
            "representative_forced_lexicalization": forced_render,
            "interpretation": "A feasible wildcard topology is not a candidate: it still needs two original coherent English insertions, a no-shortcut token-boundary audit, and exact independent validation. This bounded run deliberately stops before substituting catalogue/reversed-token strings.",
        },
        "candidate": None,
        "readability": {"status": "not_evaluated", "reason": "no lexicalized candidate exists; wildcard feasibility cannot certify English"},
        "next_operator": "Change from insertion-only geometry to one retained-span replacement plus one insertion/deletion compensation, then solve the residual on a genuinely new parent seam before lexicalization; do not resweep these cuts or widen the width range.",
    }
    OUT_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT_PATH), "tested": record["search"]["tested_topologies"], "cut_count": len(cuts), "feasible": len(results), "best": best}, indent=2))


if __name__ == "__main__":
    main()
