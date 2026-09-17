"""Independent contract check for the ten orthogonal Luna lanes.

This is deliberately a validator, not another search.  It replays the
rendered text in each selected run with an independent letter normalizer,
two-pointer comparison, and forward/reverse SHA-256 digests.  A lane is only
accepted into the matrix when it also carries provenance, novelty preflight,
and a concrete next repair in its own run artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


LANES = [
    (1, "character-level LM-constrained decoding", "runs/live-gpt2-character-decoder-preflight-20260916.json", "lm"),
    (2, "exact-tape grammatical resegmentation", "runs/exact-tape-grammatical-resegmentation-20260916-luna.json", "tape"),
    (3, "dependency-tree seam CSP", "runs/dependency-seam-csp-20260916-luna.json", "dependency"),
    (4, "agreement-carrying morphology transducer", "runs/morphology-crossword-transducer-20260916.json", "morphology"),
    (5, "CFG/Earley character intersection", "runs/cfg-earley-character-intersection-fresh-20260916.json", "cfg"),
    (6, "human-authored scene lattice with live equations", "runs/human-scene-lattice-live-equations-20260916.json", "scene"),
    (7, "semantic valency/attachment solver", "runs/semantic-valency-attachment-solver-20260916-luna.json", "valency"),
    (8, "inflectional and clitic boundary search", "runs/inflection-clitic-distinct-repair-20260916-luna.json", "clitic"),
    (9, "scalable compositional grammar without nested palindrome spans", "runs/scalable-compositional-grammar-20260916-luna.json", "scalable"),
    (10, "exact-candidate repair using semantic slot substitutions", "runs/semantic-slot-substitution-repair-20260916-luna.json", "slot"),
]


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


def exact_two_pointer(tape: str) -> tuple[bool, int | None]:
    mismatches = 0
    for left in range(len(tape) // 2):
        right = len(tape) - 1 - left
        if tape[left] != tape[right]:
            mismatches += 1
    return mismatches == 0, mismatches


def rendered_and_audit(payload: dict, kind: str) -> tuple[str, dict]:
    if kind == "lm":
        row = payload["rendered_candidates"][0]
        return row["rendered"], row
    if kind == "tape":
        row = payload["candidate"]
        return row["rendered"], row
    if kind == "dependency":
        return payload["rendered"], payload["independent_exact_audit"]
    if kind == "morphology":
        row = payload["candidates"][0]
        return row["audit"]["rendered"], row["audit"]
    if kind == "cfg":
        row = payload["candidates"][0]["audit"]
        return row["rendered"], row
    if kind == "scene":
        row = payload["candidate"]
        return row["rendered"], row["independent_validation"]
    if kind == "valency":
        return payload["rendered"], payload["audit"]
    if kind == "clitic":
        row = payload["candidate"]
        return row["rendered"], payload["independent_pointer_sha_audit"]
    if kind == "scalable":
        row = payload["intact_english_prose_candidate"]
        return row["rendered"], row["independent_checks"]
    if kind == "slot":
        row = payload["candidate"]
        return row["rendered"], row
    raise ValueError(kind)


def next_repair(payload: dict, kind: str) -> object:
    for key in ("next_repair", "next_repair_operator", "next_extension_repair", "repair", "repair_at_first_residual"):
        if key in payload:
            value = payload[key]
            if isinstance(value, dict) and "concrete_action" in value:
                return value["concrete_action"]
            return value
    if kind == "scene":
        return payload["candidate"].get("next_repair")
    raise AssertionError("missing concrete next repair")


def main() -> dict:
    rows = []
    for number, lane, relative, kind in LANES:
        payload = json.loads((ROOT / relative).read_text())
        text, source_audit = rendered_and_audit(payload, kind)
        tape = letters(text)
        exact, mismatch_count = exact_two_pointer(tape)
        forward = hashlib.sha256(tape.encode()).hexdigest()
        reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
        assert tape and text.strip(), f"lane {number} has no rendered prose"
        provenance = payload.get("provenance")
        if provenance is None and kind == "scene":
            provenance = payload["candidate"].get("provenance")
        assert isinstance(provenance, (dict, str)), f"lane {number} missing provenance"
        assert isinstance(payload.get("novelty_preflight"), dict), f"lane {number} missing novelty preflight"
        assert next_repair(payload, kind), f"lane {number} missing concrete next repair"
        assert not exact, f"lane {number} unexpectedly changed status; inspect before promotion"
        if kind == "clitic":
            assert payload["candidate"]["repeated_content_units"] == 0
            assert payload["candidate"]["word_order_symmetry"] is False
        rows.append({
            "lane": number,
            "name": lane,
            "run": relative,
            "rendered": text,
            "letters": len(tape),
            "independent_two_pointer_exact": exact,
            "mismatch_count": mismatch_count,
            "sha256_forward": forward,
            "sha256_reverse": reverse,
            "source_audit_exact": source_audit.get("exact", source_audit.get("two_pointer_exact", False)),
            "provenance_present": True,
            "novelty_preflight_present": True,
            "next_repair_present": True,
        })
    return {
        "status": "contract_validated_diagnostic_only",
        "lane_count": len(rows),
        "exact_count": sum(row["independent_two_pointer_exact"] for row in rows),
        "mechanically_admitted_count": 0,
        "rows": rows,
        "note": "Programmatic checks diagnose exactness and construction provenance; they do not certify human readability.",
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
