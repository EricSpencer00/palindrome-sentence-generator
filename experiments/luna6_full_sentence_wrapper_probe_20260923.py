"""Test one sentence at the released [0,232)/end-568 exact wrapper seam."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT_PATH = ROOT / "runs/luna6-full-sentence-wrapper-probe-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
Y = "A courier carried the witness ledger to the village before dawn."


def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())


def audit(s: str) -> dict:
    tape = letters(s)
    mismatch = next(
        (
            {"left": i, "right": len(tape) - i - 1, "left_char": tape[i], "right_char": tape[-i - 1]}
            for i in range(len(tape) // 2)
            if tape[i] != tape[-i - 1]
        ),
        None,
    )
    return {"letters": len(tape), "exact": mismatch is None, "first_mismatch": mismatch}


def main() -> None:
    from llm_palindrome.validator import is_palindrome

    raw = json.loads(PARENT_PATH.read_text())
    parent = next(row["rendered"] for row in raw["rows"] if row["id"] == "outer-causal-scene-568-working-incumbent")
    p = letters(parent)
    assert len(p) == 568 and hashlib.sha256(p.encode()).hexdigest() == PARENT_SHA256
    assert audit(parent)["exact"] and is_palindrome(parent)

    y = letters(Y)
    # For [0,232)/end-568, X=reverse(Y)+P[0:232], so the whole tape is
    # reverse(Y)+P+Y. Keep reverse(Y) contiguous: no punctuation or spacing
    # may change its letters, and no false word-level readability is implied.
    x = y[::-1] + p[:232]
    rendered = y[::-1] + parent + " " + Y
    tape = letters(rendered)
    expected = y[::-1] + p + y
    assert x == y[::-1] + p[:232]
    assert tape == expected
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    independent = audit(rendered)
    project = bool(is_palindrome(rendered))

    # This reports the obstruction rather than silently spacing character
    # reversal fragments into token-shaped units.
    reverse_surface = y[::-1]
    reverse_tokens = re.findall(r"[a-z]+", reverse_surface)
    right_tokens = re.findall(r"[a-z]+", Y.lower())
    result = {
        "experiment_id": "luna6-full-sentence-wrapper-probe-20260923",
        "status": "exact_but_rejected_unreadable_reversal_seam",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": len(p),
            "normalized_sha256": PARENT_SHA256,
            "rendered": parent,
        },
        "operator": {
            "geometry": {"replace_parent_span": [0, 232], "insert_at_parent_cut": 568},
            "identity": "X=reverse(Y)+P[0:232]; rendered tape=reverse(Y)+P+Y",
            "Y_surface": Y,
            "Y_letters": len(y),
            "X_letters": len(x),
            "predicted_letters": 568 + 2 * len(y),
            "actual_letters": len(tape),
            "growth": len(tape) - 568,
            "live_residual": "At each outer-in step, the added prefix reverse(Y) is paired with added suffix Y; the unchanged parent is centered and independently exact.",
        },
        "candidate": {
            "rendered": rendered,
            "normalized_sha256": forward,
            "reverse_sha256": reverse,
            "independent_outside_in": independent,
            "project_validator_exact": project,
            "mechanically_exact": independent["exact"] and project and forward == reverse,
        },
        "readability_audit": {
            "right_new_clause": Y,
            "right_clause_sentence_parse": "complete, grammatical sentence; narrative link to Noel is not explicit",
            "required_left_surface": reverse_surface,
            "left_after_spaces_cannot_change": ["Leon won. Wolf spots Nora."],
            "left_prefix_parse": "no coherent sentence parse identified; contiguous reversal is not readable English",
            "reverse_surface_tokens_if_naively_split": reverse_tokens,
            "right_surface_tokens": right_tokens,
            "whole_token_reversal_shortcut_check": "not admitted: any spacing that creates tokenwise reversals is forbidden; no left parse was found",
            "repeated_new_units": [],
            "reader_evidence": False,
            "status": "not a reader-worthy candidate; exactness does not certify readability",
        },
        "lexical_obstruction": {
            "exact_tape_obligation": y[::-1],
            "surface_before_parent": reverse_surface,
            "reason": "This one complete right sentence has no coherent full-sentence parse under its forced character reversal. Keeping it contiguous makes the failure visible; adding token boundaries cannot repair the letters and risks a forbidden word-mirror shortcut.",
            "next_operator": {
                "geometry": {"replace_centered_palindromic_span": [232, 336], "preserve_outer_reflection_shells": [[0, 232], [336, 568]]},
                "identity": "replace M=P[232:336] by reverse(Z)+M+Z, retaining L=P[0:232] and reverse(L); new tape L+reverse(Z)+M+Z+reverse(L)",
                "change": "Move the authored event to the centered 104-letter span instead of wrapping the whole parent. This gives the new clause the parent’s interior discourse context at both boundaries and is not the [0,232)/end-568 wrapper topology.",
                "preflight": {
                    "exact_coordinate_signature": "no hit for [232,336] in tracked runs/experiments/docs/data",
                    "operator_collision": "docs/EXPERIMENT-NOVELTY-REGISTRY.md records reversible-semantic-wrappers: fresh semantic centers surrounded by character-pair wrapper primitives. This is the same construction family, so the central wrap is retired despite the new coordinates.",
                    "decision": "do not run a lexical realization under this duplicate operator",
                },
                "next_distinct_action": "Return to the goal owner for a materially different construction family; do not rename another character-pair wrapper or seam coordinate as novelty.",
            },
        },
        "provenance": {
            "Y": Y,
            "source": "newly authored single sentence for this seam test; not imported from a catalogue",
            "candidate_text_saved": True,
        },
    }
    assert independent["exact"] and project and forward == reverse
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"path": str(OUTPUT_PATH), "letters": len(tape), "growth": len(tape)-568, "exact": True, "sha256": forward, "reverse_prefix": reverse_surface}, indent=2))


if __name__ == "__main__":
    main()
