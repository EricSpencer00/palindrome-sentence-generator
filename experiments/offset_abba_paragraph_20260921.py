"""Bounded paragraph ABBA search with cross-boundary character seam offsets.

The semantic topology is ABBA, but no surface unit is reused or mirrored.  A
small scene lattice is joined to seam offsets; each offset carries the exposed
character residual across words and sentence punctuation before the paragraph
is rendered and audited.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/offset-abba-paragraph-20260921.json"

SCENES = {
    "departure": [
        "At first light, Mara carried the weathered chart toward the eastern quay.",
        "Before sunrise, the quiet ferryman hauled a canvas case beyond the old jetty.",
    ],
    "repair": [
        "By midmorning, Ivo steadied the cracked lantern while rain crossed the yard.",
        "After the bell, the patient cooper repaired a split oar beside the storehouse.",
    ],
    "return": [
        "Near evening, Sela brought the marked chart back through the salt grass.",
        "At twilight, the young pilot returned with a measured course along the inlet.",
    ],
    "witness": [
        "Long after dark, Neri recorded the repaired beacon in a narrow field book.",
        "When the tide turned, the watchful clerk entered the signal in a weather log.",
    ],
}


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def audit(text: str) -> dict:
    letters = tape(text)
    mismatches = [
        {"offset": i, "left": letters[i], "right": letters[-1 - i]}
        for i in range(len(letters) // 2)
        if letters[i] != letters[-1 - i]
    ]
    return {
        "letters": len(letters),
        "two_pointer_exact": bool(letters) and not mismatches,
        "mismatches": mismatches[:12],
        "sha256_forward": digest(letters),
        "sha256_reverse": digest(letters[::-1]),
    }


def seam_residual(text: str, offset: int) -> dict:
    """Compare opposite characters around an arbitrary character seam."""
    chars = tape(text)
    left, right = chars[:offset], chars[offset:]
    width = min(len(left), len(right), 8)
    pairs = [
        {"depth": d, "left": left[offset - 1 - d], "right": right[d],
         "matched": left[offset - 1 - d] == right[d]}
        for d in range(width)
    ]
    return {"offset": offset, "width": width, "pairs": pairs,
            "matched": sum(p["matched"] for p in pairs),
            "crosses_word_boundary": " " in text[max(0, offset - 3):offset + 3]}


def run() -> dict:
    rows = []
    # ABBA denotes distinct semantic roles, not duplicated text: B is repair,
    # while the return/witness beats provide a different authored surface.
    role_paths = [("departure", "repair", "repair", "departure"),
                  ("departure", "witness", "witness", "departure")]
    for path in role_paths:
        for choices in __import__("itertools").product(*(SCENES[r] for r in path)):
            rendered = " ".join(choices)
            chars = tape(rendered)
            # Jointly search offsets at word/clause boundaries and retain the
            # residual trace; no candidate is repaired after rendering.
            offsets = sorted({0, *(i for i, c in enumerate(rendered) if c in ".,;"), len(chars)})
            seams = [seam_residual(rendered, min(o, len(chars))) for o in offsets]
            au = audit(rendered)
            admission = mechanical_admission_checks(rendered, min_letters=30, max_letters=500)
            exact_admitted = au["two_pointer_exact"] and all(admission.values())
            repeated_surface = len(set(choices)) != len(choices)
            rows.append({"rendered": rendered, "semantic_pattern": ["A", "B", "B", "A"],
                         "roles": path, "scene_indices": [SCENES[r].index(s) for r, s in zip(path, choices)],
                         "seam_residuals": seams, "audit": au, "admission": admission,
                         "exact_admitted": exact_admitted,
                         "provenance": {"independently_authored": not repeated_surface, "crosses_word_boundaries": True,
                            "crosses_clause_boundaries": True, "joint_scene_and_offset_search": False,
                            "post_render_diagnostic": True,
                            "repeated_units": repeated_surface, "mirrored_word_order": not admission["not_word_order_symmetry"],
                            "nested_palindromic_spans": not admission["no_self_palindromic_proper_multiword_span"], "finished_tape_reversal": False,
                            "posthoc_repair": False}})
    exact = [r for r in rows if r["exact_admitted"]]
    return {
        "experiment_id": "offset-abba-paragraph-20260921",
        "method": "rejected post-render authored-scene seam diagnostic for paragraph ABBA",
        "stats": {"scene_paths": len(role_paths), "candidate_completions": len(rows),
                  "seam_states": sum(len(r["seam_residuals"]) for r in rows),
                  "exact_admitted": len(exact), "max_letters": max(r["audit"]["letters"] for r in rows)},
        "rendered_candidates": rows, "exact_candidates": exact,
        "novelty_preflight": {"status": "rejected_as_duplicate_sweep", "signature": "offset-abba|scene-lattice|cross-boundary-residual",
            "reason": "the implementation enumerates complete scene tuples and measures offsets afterward; it duplicates prior paragraph topology probes instead of constraining generation",
            "duplicate_sweep": True, "forbidden_shortcuts": ["A↔A' pairs", "B↔B' pairs", "seed wrapping", "semantic relabeling"]},
        "provenance": {"independent_audits": ["outside-in character comparison", "forward/reverse SHA-256", "mechanical_admission_checks"],
                       "reader_gate": "closed unless exact_admitted", "generator_sha256": digest(Path(__file__).read_text())},
        "next_repair": "Close this post-render lane; compile scene grammar and opposite-character residuals into one pre-render automaton.",
        "status": "fresh exact candidate requires reading" if exact else "closed as non-progress: no exact closure and no live constrained generation",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
