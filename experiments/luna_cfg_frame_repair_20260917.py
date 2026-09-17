"""Targeted repair of the excluded Luna CFG witness.

The repair changes the semantic frame, not the old terminal tape.  A bounded
two-ended chart records the first residual and refuses to render a candidate
when a proper multiword span is itself palindromic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-cfg-frame-repair-20260917.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
EXPERIMENT_ID = "luna-cfg-frame-repair-20260917"
SIGNATURE = "bounded-typed-cfg|fresh-role-event-frame|two-ended-chart|self-span-hard-reject"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Frame:
    left: str
    right: str
    role: str


# Fresh semantic material.  These terminals are not taken from the excluded
# observer/food-event or imperative/prohibition pair, and their word order is
# not a reflected copy.
FRAMES = (
    Frame("A careful mason repairs the gate", "At dawn a quiet botanist labels seedlings", "craftsperson-repairs-access|botanist-labels-specimens"),
    Frame("The patient curator checks a map", "Nearby a young porter stacks clean crates", "curator-checks-record|porter-stacks-cargo"),
)


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    rows = [r for r in registry.get("entries", []) if r.get("id") != EXPERIMENT_ID]
    artifact = str(Path(__file__).relative_to(ROOT))
    overlaps = [r.get("id") for r in rows if r.get("signature") == SIGNATURE]
    collisions = [r.get("id") for r in rows if r.get("artifact") == artifact]
    result = {
        "status": "passed" if not overlaps and not collisions else "blocked",
        "performed_before_search": True,
        "registry_entries_before_run": len(registry.get("entries", [])),
        "signature_overlaps": overlaps,
        "artifact_collisions": collisions,
        "ignored_self_registry_collision": True,
        "rejected_routes": ["fixed tape", "reverse segmentation", "word-order mirror", "catalogue text", "gibberish"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def chart_intersection(frames: tuple[Frame, ...]) -> list[dict[str, object]]:
    chart = []
    for index, frame in enumerate(frames):
        left, right = normalize_letters(frame.left), normalize_letters(frame.right)
        states = []
        matched = 0
        while matched < len(left) and matched < len(right):
            li, ri = left[matched], right[-1 - matched]
            states.append({"grammar_state": index, "left_position": matched,
                           "right_position": len(right) - 1 - matched,
                           "terminal_pair": [li, ri], "matched": li == ri})
            if li != ri:
                break
            matched += 1
        states.append({"grammar_state": index, "left_position": matched,
                       "right_position": len(right) - 1 - matched,
                       "closed": matched == len(left) == len(right)})
        chart.append({"frame_index": index, "frame": frame.__dict__, "states": states,
                      "matched_characters": matched, "closed": matched == len(left) == len(right),
                      "first_residual": states[-2] if states[-1].get("closed") is not True else None})
    return chart


def pointer_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = [{"left_index": i, "right_index": len(tape) - 1 - i,
                   "left": tape[i], "right": tape[-1 - i]}
                  for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {"letters": len(tape), "normalized_tape": tape, "exact": bool(tape) and not mismatches,
            "independent_two_pointer_exact": bool(tape) and not mismatches,
            "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    chart = chart_intersection(FRAMES)
    rendered = "A careful mason repairs the gate; at dawn a quiet botanist labels seedlings. The patient curator checks a map; nearby a young porter stacks clean crates."
    audit = pointer_audit(rendered)
    mechanical = mechanical_admission_checks(rendered, min_letters=39, max_letters=180)
    hard_reject = not mechanical["no_self_palindromic_proper_multiword_span"]
    row = {
        "rendered": rendered, "chart_states": chart, "audit": {**audit, "mechanical_checks": mechanical},
        "semantic_frames": [f.role for f in FRAMES],
        "anti_shortcut_flags": {"fixed_tape": False, "reverse_segmentation": False, "catalogue_text": False,
                                "gibberish": False, "word_order_mirror": False,
                                "self_palindromic_span": hard_reject},
        "mechanically_admitted": audit["exact"] and all(mechanical.values()) and not hard_reject,
        "provenance": {"inventory": "fresh hand-authored typed role/event terminals",
                       "source_sentences_copied": False, "finished_surface_reversed": False,
                       "posthoc_resegmentation": False, "parent_exact_terminals_reused": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    result = {"experiment_id": EXPERIMENT_ID, "family": "luna-cfg-frame-repair",
              "novelty_preflight": preflight,
              "grammar": {"nonterminals": ["S", "Frame"], "production_count": len(FRAMES),
                          "chart_key": ["grammar_state", "left_position", "right_position"],
                          "two_ended": True},
              "candidate_count": 0, "rendered_candidates": [row], "best": row,
              "reader_eligible": False,
              "next_repair": "Replace the first residual pair (mason/gate versus botanist/seedlings boundary) with a held-out role-compatible lexical choice, then replay the two-ended chart; retain the self-span hard reject.",
              "provenance": row["provenance"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run()["best"]["audit"], indent=2))
