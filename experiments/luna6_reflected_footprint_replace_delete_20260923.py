#!/usr/bin/env python3
"""One expanded-sentence replacement and one deletion inside its old reflection footprint."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-reflected-footprint-replace-delete-20260923.json"
EXPECTED_PARENT = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OLD_SCENE = "Sara, did I live? Nora, I saw desserts."
NEW_SCENE = (
    "After rain, Sara followed one lantern through orchard paths and found Nora "
    "beside locked gates; together they carried maps home safely before dawn."
)
DELETE_SENTENCE = "Stressed was I, Aron."


def tape(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def outside_in(text: str) -> dict:
    t = tape(text)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {
        "letters": len(t), "exact": mismatch is None,
        "first_mismatch": ({"offset": mismatch[0], "left": mismatch[1], "right": mismatch[2],
                            "right_offset": len(t)-1-mismatch[0]} if mismatch else None),
    }


def project(text: str) -> dict:
    from llm_palindrome.validator import is_palindrome
    return {"exact": bool(is_palindrome(text))}


def main() -> None:
    parent_doc = json.loads(PARENT.read_text())
    parent = parent_doc["rows"][0]["rendered"]
    p = tape(parent)
    if sha(p) != EXPECTED_PARENT or not outside_in(parent)["exact"] or not project(parent)["exact"]:
        raise SystemExit("Pinned parent provenance/exactness mismatch")
    if parent.count(OLD_SCENE) != 1 or parent.count(DELETE_SENTENCE) != 1:
        raise SystemExit("Source scene uniqueness precondition failed")

    replaced = parent.replace(OLD_SCENE, NEW_SCENE)
    rendered = replaced.replace(
        "Desserts I saw, Leon. " + DELETE_SENTENCE + " Evil I did, Aras.",
        "Desserts I saw, Leon. Evil I did, Aras.",
    )
    if rendered == replaced:
        raise SystemExit("Deletion sentence not found in expected retained context")
    child = tape(rendered)

    a, b = 204, 232
    repl_width = len(tape(NEW_SCENE))
    delete_a, delete_b = 336, 352
    delete_shifted = delete_a + repl_width - (b-a)
    center_twice = len(child)
    overlap_start = max(a, center_twice - (a + repl_width))
    overlap_end = min(a + repl_width, center_twice - a)
    internal_pairs = max(0, (overlap_end - overlap_start) // 2)
    forced_prefix_end = max(a, overlap_start)
    forced_prefix = "".join(child[len(child)-1-i] for i in range(a, forced_prefix_end))
    proposed_prefix = tape(NEW_SCENE)[:len(forced_prefix)]

    token_list = [w.lower() for w in __import__("re").findall(r"[A-Za-z]+", NEW_SCENE)]
    reverse_pairs = [[x, y] for i, x in enumerate(token_list) for y in token_list[i+1:] if x[::-1] == y]
    self_pal = [w for w in token_list if w == w[::-1]]
    repeats = sorted({w for w in token_list if token_list.count(w) > 1})
    audit = outside_in(rendered)
    if audit["exact"]:
        raise SystemExit("Unexpected exact closure; re-audit before retaining")

    record = {
        "experiment_id": "luna6-reflected-footprint-replace-delete-20260923",
        "status": "rejected_exactness_residual",
        "method": "replace one complete sentence with a fresh event paragraph and delete one complete sentence from the old reflection footprint",
        "novelty_preflight": {
            "parent": str(PARENT.relative_to(ROOT)),
            "parent_sha256": EXPECTED_PARENT,
            "exact_geometry_checked": {"replacement": [a, b], "deletion": [delete_a, delete_b]},
            "coordinate_query": "No exact [204,232)/[336,352) geometry hit in tracked runs, experiments, docs, or the experiment novelty registry.",
            "adjacent_prior_geometry": [
                "[204,220)/[348,364) attachment-pair residual",
                "[222,248)/[320,346) partial-word semantic scene",
            ],
            "textual_overlap_note": "The parent phrases occur in historical tape; their presence as inherited text is not a matching edit. This run changes the full [204,232) sentence span and deletes [336,352).",
        },
        "provenance": {
            "parent_letters": len(p), "parent_normalized_sha256": sha(p),
            "parent_outside_in": outside_in(parent), "parent_project_validator": project(parent),
            "replacement_source": OLD_SCENE, "replacement_parent_span": [a, b],
            "replacement_text": NEW_SCENE, "replacement_letters": repl_width,
            "deletion_source": DELETE_SENTENCE, "deletion_parent_span": [delete_a, delete_b],
            "deletion_postreplacement_span": [delete_shifted, delete_shifted + (delete_b-delete_a)],
            "net_growth": repl_width - (b-a) - (delete_b-delete_a),
        },
        "rendered": rendered,
        "candidate": {
            "letters": len(child), "normalized_sha256": sha(child),
            "outside_in": audit, "project_validator": project(rendered), "exact": False,
        },
        "seam_ledger": {
            "new_scene_tape_span": [a, a+repl_width],
            "final_tape_length": len(child),
            "candidate_reflection_overlap": [overlap_start, overlap_end],
            "variable_variable_pairs": internal_pairs,
            "new_scene_prefix_paired_to_retained_text": {
                "span": [a, forced_prefix_end],
                "required_from_opposite_retained_tape": forced_prefix,
                "proposed_prefix": proposed_prefix,
                "first_mismatch": audit["first_mismatch"],
                "interpretation": "The cursor fails at the first new character, before the three internal variable-variable pairs; the realization does not close the retained-text residual.",
            },
        },
        "boundary_audit": {
            "new_event_tokens": token_list,
            "whole_token_reversal_pairs": reverse_pairs,
            "self_palindromic_tokens": self_pal,
            "repeated_tokens": repeats,
            "shortcut_free_new_event": not (reverse_pairs or self_pal or repeats),
        },
        "readability": {"status": "not_reader_evaluated", "reason": "The tape is not exact and the result is not reader-eligible."},
        "next_operator": "Change the topology to a second independently lexicalized insertion at the opposite live owner, so the first character at offset 204 is paired variable-to-variable instead of being forced by retained text; preflight that two-edit geometry before drafting, and do not phrase-swap this sentence.",
    }
    OUTPUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "letters": len(child), "sha256": sha(child),
                      "first_mismatch": audit["first_mismatch"], "variable_variable_pairs": internal_pairs,
                      "required_prefix": forced_prefix}, indent=2))


if __name__ == "__main__":
    main()
