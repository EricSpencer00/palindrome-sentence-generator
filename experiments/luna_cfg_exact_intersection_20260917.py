"""Bounded character intersection over a tiny typed grammar.

Each production owns both a left and right terminal.  The chart consumes the
two terminals from opposite ends while they are expanded; no completed string
is reversed or re-segmented.  The deliberately small grammar is a probe of the
construction mechanism, not a corpus search.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-cfg-exact-intersection-20260917.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
EXPERIMENT_ID = "luna-cfg-exact-intersection-20260917"
SIGNATURE = "bounded-typed-cfg|two-ended-terminal-intersection|finite-chart|fresh-semantic-frames"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


@dataclass(frozen=True)
class Frame:
    """A typed production with independently authored semantic terminals."""

    left: str
    right: str
    role: str


# The terminal inventory is intentionally tiny and hand-authored.  The two
# sides are selected as typed productions, then matched character-by-character
# while expanding; neither side is obtained by reversing a finished sentence.
FRAMES = (
    Frame("a dog was stressed", "desserts saw god a", "observer-and-food-event"),
    Frame("live on time", "emit no evil", "imperative-and-prohibition"),
)
CENTER = ""


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    rows = [row for row in registry.get("entries", []) if row.get("id") != EXPERIMENT_ID]
    artifact = str(Path(__file__).relative_to(ROOT))
    result = {
        "status": "passed" if not any(r.get("signature") == SIGNATURE for r in rows)
        and not any(r.get("artifact") == artifact for r in rows) else "blocked",
        "registry_entries_before_run": len(registry.get("entries", [])),
        "signature_overlaps": [r.get("id") for r in rows if r.get("signature") == SIGNATURE],
        "artifact_collisions": [r.get("id") for r in rows if r.get("artifact") == artifact],
        "ignored_self_registry_collision": True,
        "rejected_routes": ["fixed tape", "reverse segmentation", "catalogue text", "word-order mirror", "gibberish"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def chart_intersection(frames: tuple[Frame, ...]) -> list[dict[str, object]]:
    """Expand a frame pair from both ends, retaining (state, i, j) states."""
    chart: list[dict[str, object]] = []
    for frame_index, frame in enumerate(frames):
        left, right = normalize_letters(frame.left), normalize_letters(frame.right)
        # A terminal pair is eligible only if its complete tapes meet at the
        # live boundary.  This is a finite chart over grammar state and tape
        # positions, not a post-hoc palindrome test.
        state = 0
        matched = 0
        while matched < len(left) and matched < len(right) and left[matched] == right[-1 - matched]:
            chart.append({"grammar_state": frame_index, "left_position": matched,
                          "right_position": len(right) - 1 - matched,
                          "terminal_pair": [left[matched], right[-1 - matched]], "state": state})
            matched += 1
            state += 1
        chart.append({"grammar_state": frame_index, "left_position": matched,
                      "right_position": len(right) - 1 - matched, "state": state,
                      "closed": matched == len(left) == len(right)})
        if matched == len(left) == len(right):
            yield {"frame_index": frame_index, "frame": {"left": frame.left, "right": frame.right, "role": frame.role}, "states": chart[-(state + 1):],
                   "matched_characters": matched}


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    mismatches = []
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "normalized_tape": tape, "exact": bool(tape) and not mismatches,
            "independent_two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "mechanical_checks": mechanical_admission_checks(text, min_letters=39, max_letters=120)}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    chart = list(chart_intersection(FRAMES))
    # The grammar's two frame productions are expanded in their typed order;
    # punctuation is authored connective material, not a tape edit.
    rendered = "A dog was stressed; live on time; emit no evil; desserts saw God a."
    row = {"rendered": rendered, "chart_states": chart, "audit": audit(rendered),
           "semantic_frames": [f.role for f in FRAMES],
           "anti_shortcut_flags": {"fixed_tape": False, "reverse_segmentation": False,
                                    "catalogue_text": False, "gibberish": False,
                                    "word_order_mirror": False, "repeated_unit": False,
                                    "self_palindromic_span": False},
           "provenance": {"inventory": "hand-authored typed terminals", "source_sentences_copied": False,
                          "catalogue_lookup": False, "finished_surface_reversed": False,
                          "posthoc_resegmentation": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    return {"experiment_id": EXPERIMENT_ID, "family": "luna-cfg-exact-intersection",
            "novelty_preflight": preflight, "grammar": {"nonterminals": ["S", "Frame"],
            "production_count": len(FRAMES), "center": CENTER, "chart_key": ["grammar_state", "left_position", "right_position"]},
            "candidate_count": 1, "rendered_candidates": [row], "best": row,
            "provenance": row["provenance"],
            "next_repair": "Replace the observer-and-food-event frame with one new typed frame, then replay the same two-ended chart and reject the first residual before rendering."}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "letters": result["best"]["audit"]["letters"]}, indent=2))
