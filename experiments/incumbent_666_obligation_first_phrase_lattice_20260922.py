"""Run an obligation-first bounded phrase lattice on two real 666 seams.

The lattice derives the required reciprocal prefix before expanding.  Each
expansion is a finite, complete clause from the authored grammar and carries
the live discourse/grammar state needed to reject fragments, catalogue-like
strings, and boundary frame repetition.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    independent_audit,
    normalize,
    validate_frontier_entry,
)


PARENT = ROOT / "runs" / "incumbent-666-central-mini-scene-comparison-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-obligation-first-phrase-lattice-20260922.json"
PARENT_ID = "central-mini-scene-comparison-leon-noel-666"
PARENT_SHA256 = "3951b9449ed3ab28f55d9798e344dfaf3123035f5f07c9dffdf0047bed0e1d79"

PRIMARY = {
    "normalized_left": (197, 281),
    "normalized_right": (385, 469),
    "raw_left": (263, 384),
    "raw_right": (520, 644),
}
SWITCH = {
    "normalized_left": (127, 197),
    "normalized_right": (469, 539),
    "raw_left": (171, 263),
    "raw_right": (644, 736),
}

BEAM_WIDTH = 16
MAX_EXPANSIONS = 128
MAX_CLAUSES_PER_SIDE = 4
MAX_BACKTRACKS_PER_CURSOR = 2


def clause(
    ident: str,
    text: str,
    subject: str,
    verb: str,
    objects: tuple[str, ...],
    punctuation: str,
    relation: str | None = None,
    phase: str = "statement",
) -> dict[str, object]:
    return {
        "id": ident,
        "text": text,
        "normalized": normalize(text),
        "subject": subject,
        "verb": verb,
        "objects": list(objects),
        "frame": f"{subject.lower()}|{verb.lower()}",
        "punctuation": punctuation,
        "relation": relation,
        "phase": phase,
        "complete_finite": True,
    }


# This is intentionally a small authored phrase vocabulary, not a Cartesian
# sentence generator.  The first seam's required prefixes select only the
# old shell's first four complete clauses; the switched seam selects its own
# first four.  Other entries document the finite grammar's valid alternatives
# without allowing arbitrary word-level filler.
PHRASES = (
    clause("now_noel_did_live", "Now, Noel, did I live?", "I", "did live", ("Noel",), "?", "temporal", "question"),
    clause("nora_saw_noel_live", "Nora saw Noel live.", "Nora", "saw", ("Noel",), "."),
    clause("noel_i_sit", "Noel, I sit.", "I", "sit", ("Noel",), "."),
    clause("pat_notes", "Pat notes.", "Pat", "notes", (), "."),
    clause("mara_saw_god", "Mara saw God.", "Mara", "saw", ("God",), "."),
    clause("sara_did_live", "Sara, did I live?", "I", "did live", ("Sara",), "?", "temporal", "question"),
    clause("nora_saw_desserts", "Nora, I saw desserts.", "I", "saw", ("desserts",), "."),
    clause("stressed_was_aron", "Stressed was I, Aron.", "I", "was", ("Aron",), "."),
    clause("evil_did_aras", "Evil I did, Aras.", "I", "did", ("Aras",), "."),
    clause("dog_was_aram", "Dog was Aram.", "Dog", "was", ("Aram",), "."),
    clause("seton_tap", "Seton, tap.", "you", "tap", ("Seton",), ".", phase="imperative"),
    clause("tis_i_leon", "'Tis I, Leon.", "I", "'tis", ("Leon",), "."),
    clause("evil_leon_was_aron", "Evil Leon was Aron.", "Evil Leon", "was", ("Aron",), "."),
    clause("evil_leon_was_aras", "Evil Leon was Aras.", "Evil Leon", "was", ("Aras",), "."),
    clause("evil_did_leon_won", "Evil I did, Leon won.", "I", "did", ("Leon",), "."),
    clause("mara_sees_nadia", "Mara sees Nadia.", "Mara", "sees", ("Nadia",), "."),
    clause("nadia_saw_noel_live", "Nadia saw Noel live.", "Nadia", "saw", ("Noel",), "."),
    clause("mara_stops_nadia", "Mara stops Nadia.", "Mara", "stops", ("Nadia",), "."),
    clause("nora_sees_aram", "Nora sees Aram.", "Nora", "sees", ("Aram",), "."),
    clause("sara_saw_noel_live", "Sara saw Noel live.", "Sara", "saw", ("Noel",), "."),
    clause("nora_sees_mara", "Nora sees Mara.", "Nora", "sees", ("Mara",), "."),
    clause("nadia_stops_aram", "Nadia stops Aram.", "Nadia", "stops", ("Aram",), "."),
    clause("mara_sees_aron", "Mara sees Aron.", "Mara", "sees", ("Aron",), "."),
    clause("aidan_spots_aram", "Aidan spots Aram.", "Aidan", "spots", ("Aram",), "."),
    clause("evil_leon_was_aidan", "Evil Leon was Aidan.", "Evil Leon", "was", ("Aidan",), "."),
)


def frames_in_parent(rendered: str) -> set[str]:
    # Parent frame extraction is deliberately simple and conservative; the
    # authored phrase metadata remains the source of truth for candidate state.
    verbs = {str(item["verb"]) for item in PHRASES}
    result: set[str] = set()
    for sentence in re.split(r"[.!?;]+", rendered):
        words = sentence.strip().split()
        for index, word in enumerate(words):
            cleaned = word.strip(",:'\"“”").lower()
            if cleaned in verbs and index > 0:
                subject = " ".join(words[:index]).strip(",:'\"“”").lower()
                result.add(f"{subject}|{cleaned}")
                break
    return result


def boundary_context(rendered: str, raw_window: tuple[int, int]) -> dict[str, object]:
    before = rendered[max(0, raw_window[0] - 80) : raw_window[0]]
    after = rendered[raw_window[1] : raw_window[1] + 80]
    return {
        "before_raw": before,
        "after_raw": after,
        "before_normalized_tail": normalize(before)[-24:],
        "after_normalized_prefix": normalize(after)[:24],
    }


def clause_matches_prefix(phrase: dict[str, object], residual: str) -> bool:
    phrase_tape = str(phrase["normalized"])
    required_prefix = residual[: min(3, len(phrase_tape))]
    return bool(required_prefix) and phrase_tape.startswith(required_prefix)


def matching_prefix_length(phrase: dict[str, object], residual: str) -> int:
    emitted = str(phrase["normalized"])
    limit = min(len(emitted), len(residual))
    for cursor in range(limit):
        if emitted[cursor] != residual[cursor]:
            return cursor
    return limit


def compare_emission(emission: str, obligation: str, owner: str) -> dict[str, object]:
    emitted = normalize(emission)
    limit = min(len(emitted), len(obligation))
    for cursor in range(limit):
        if emitted[cursor] != obligation[cursor]:
            return {
                "exact": False,
                "cursor": cursor,
                "expected": obligation[cursor],
                "emitted": emitted[cursor],
                "residual": obligation[cursor:],
                "owner": owner,
                "reason": "character_contradiction",
            }
    if len(emitted) > len(obligation):
        return {
            "exact": False,
            "cursor": len(obligation),
            "expected": None,
            "emitted": emitted[len(obligation)],
            "residual": "",
            "owner": owner,
            "reason": "emission_overrun",
        }
    residual = obligation[len(emitted) :]
    return {
        "exact": not residual,
        "cursor": len(emitted),
        "expected": None,
        "emitted": None,
        "residual": residual,
        "owner": owner,
        "reason": "closed" if not residual else "prefix_consumed",
    }


def initial_state(side: str, obligation: str, context: dict[str, object], neighbor_frames: set[str]) -> dict[str, object]:
    return {
        "side": side,
        "cursor": 0,
        "residual": obligation,
        "clause_owner": side,
        "subject_stack": [],
        "object_stack": [],
        "sentence_phase": "sentence_start",
        "punctuation": None,
        "active_discourse_entity": None,
        "pending_causal_temporal_relation": None,
        "used_clause_ids": [],
        "used_frame_set": [],
        "neighboring_boundary_context": context,
        "neighboring_frames": sorted(neighbor_frames),
        "trace": [],
        "grammar_rejections": [],
        "backtracks_by_cursor": {},
    }


def state_score(state: dict[str, object]) -> tuple[int, int, int]:
    return (int(state["cursor"]), len(state["used_clause_ids"]), -len(state["grammar_rejections"]))


def expand_side(
    side: str,
    obligation: str,
    context: dict[str, object],
    neighbor_frames: set[str],
) -> dict[str, object]:
    beam = [initial_state(side, obligation, context, neighbor_frames)]
    expansions = 0
    prefix_matches = 0
    grammar_rejections = 0
    backtracks: defaultdict[int, int] = defaultdict(int)
    obstruction: dict[str, object] | None = None
    completed: list[dict[str, object]] = []
    best_seen = beam[0]

    def remember_obstruction(candidate: dict[str, object]) -> None:
        nonlocal obstruction
        if obstruction is None or int(candidate["cursor"]) > int(obstruction["cursor"]):
            obstruction = candidate

    while beam and expansions < MAX_EXPANSIONS:
        next_beam: list[dict[str, object]] = []
        for state in sorted(beam, key=state_score, reverse=True)[:BEAM_WIDTH]:
            if len(state["used_clause_ids"]) >= MAX_CLAUSES_PER_SIDE:
                remember_obstruction({
                    "cursor": state["cursor"],
                    "expected": state["residual"][0] if state["residual"] else None,
                    "emitted": None,
                    "residual": state["residual"],
                    "owner": side,
                    "reason": "max_clauses_per_side_with_nonempty_residual",
                    "grammar_state": state,
                })
                if not state["residual"]:
                    completed.append(state)
                continue

            matching = [phrase for phrase in PHRASES if clause_matches_prefix(phrase, str(state["residual"]))]
            matching.sort(
                key=lambda phrase: matching_prefix_length(phrase, str(state["residual"])),
                reverse=True,
            )
            if not matching:
                remember_obstruction({
                    "cursor": state["cursor"],
                    "expected": state["residual"][0] if state["residual"] else None,
                    "emitted": None,
                    "residual": state["residual"],
                    "owner": side,
                    "reason": "no_grammar_phrase_matches_required_next_1_to_3",
                    "grammar_state": state,
                })
                continue

            for phrase in matching:
                if expansions >= MAX_EXPANSIONS:
                    break
                cursor = int(state["cursor"])
                # One first-choice expansion plus at most two retries is the
                # advertised backtrack bound.
                if backtracks[cursor] > MAX_BACKTRACKS_PER_CURSOR:
                    continue
                backtracks[cursor] += 1
                expansions += 1
                prefix_matches += 1
                phrase_id = str(phrase["id"])
                frame = str(phrase["frame"])
                shared = set(state["subject_stack"]) | set(state["object_stack"])
                links = shared & ({str(phrase["subject"])} | set(phrase["objects"]))
                rejection: str | None = None
                if phrase_id in state["used_clause_ids"]:
                    rejection = "repeated_clause"
                elif frame in state["used_frame_set"]:
                    rejection = "repeated_frame"
                elif frame in neighbor_frames:
                    rejection = "repeated_neighbor_frame"
                elif state["used_clause_ids"] and not links and not phrase["relation"]:
                    rejection = "disconnected_catalogue_clause"
                elif not phrase["complete_finite"]:
                    rejection = "incomplete_or_fragment"

                trace = {
                    "cursor": cursor,
                    "phrase_id": phrase_id,
                    "emission": phrase["text"],
                    "required_prefix": str(state["residual"])[:3],
                    "owner": side,
                    "grammar_state": {
                        "subject": phrase["subject"],
                        "objects": phrase["objects"],
                        "frame": frame,
                        "links": sorted(links),
                        "sentence_phase": phrase["phase"],
                        "punctuation": phrase["punctuation"],
                        "pending_causal_temporal_relation": phrase["relation"],
                    },
                }
                if rejection:
                    grammar_rejections += 1
                    trace["rejected"] = rejection
                    state.setdefault("grammar_rejections", []).append(trace)
                    remember_obstruction({
                        "cursor": cursor,
                        "expected": str(state["residual"])[0] if state["residual"] else None,
                        "emitted": str(phrase["normalized"])[0] if phrase["normalized"] else None,
                        "residual": state["residual"],
                        "owner": side,
                        "reason": rejection,
                        "grammar_state": state,
                    })
                    continue

                comparison = compare_emission(str(phrase["text"]), str(state["residual"]), side)
                trace["comparison"] = comparison
                if comparison["reason"] not in {"prefix_consumed", "closed"}:
                    remember_obstruction({
                        "cursor": cursor + int(comparison["cursor"]),
                        "expected": comparison["expected"],
                        "emitted": comparison["emitted"],
                        "residual": comparison["residual"],
                        "owner": side,
                        "reason": comparison["reason"],
                        "grammar_state": state,
                    })
                    continue

                new_pending = phrase["relation"]
                if links and state["pending_causal_temporal_relation"]:
                    new_pending = None
                child = dict(state)
                child["cursor"] = cursor + len(str(phrase["normalized"]))
                child["residual"] = str(state["residual"])[len(str(phrase["normalized"])) :]
                child["subject_stack"] = list(state["subject_stack"]) + [str(phrase["subject"])]
                child["object_stack"] = list(state["object_stack"]) + list(phrase["objects"])
                child["sentence_phase"] = str(phrase["phase"])
                child["punctuation"] = phrase["punctuation"]
                child["active_discourse_entity"] = (phrase["objects"] or [phrase["subject"]])[-1]
                child["pending_causal_temporal_relation"] = new_pending
                child["used_clause_ids"] = list(state["used_clause_ids"]) + [phrase_id]
                child["used_frame_set"] = list(state["used_frame_set"]) + [frame]
                child["trace"] = list(state["trace"]) + [trace]
                child["grammar_rejections"] = list(state["grammar_rejections"])
                child["backtracks_by_cursor"] = dict(backtracks)
                if state_score(child) > state_score(best_seen):
                    best_seen = child
                if not child["residual"]:
                    completed.append(child)
                else:
                    next_beam.append(child)
        beam = sorted(next_beam, key=state_score, reverse=True)[:BEAM_WIDTH]

    best = max(completed + beam + [best_seen], key=state_score)
    # Prefer the grammar obstruction on the deepest surviving prefix over a
    # shallower character mismatch from a discarded sibling branch.
    relevant_rejections = [
        item for item in best.get("grammar_rejections", [])
        if int(item["cursor"]) >= int(best["cursor"])
    ]
    if relevant_rejections and best["residual"]:
        item = max(relevant_rejections, key=lambda value: int(value["cursor"]))
        emitted = normalize(str(item["emission"]))
        obstruction = {
            "cursor": best["cursor"],
            "expected": str(best["residual"])[0],
            "emitted": emitted[0] if emitted else None,
            "residual": best["residual"],
            "owner": side,
            "reason": item["rejected"],
            "grammar_state": best,
        }
    if obstruction is None:
        obstruction = {
            "cursor": best["cursor"],
            "expected": best["residual"][0] if best["residual"] else None,
            "emitted": None,
            "residual": best["residual"],
            "owner": side,
            "reason": "bounded_search_exhausted",
            "grammar_state": best,
        }
    return {
        "side": side,
        "obligation_length": len(obligation),
        "obligation_prefix": obligation[:24],
        "beam_width": BEAM_WIDTH,
        "max_expansions": MAX_EXPANSIONS,
        "max_clauses_per_side": MAX_CLAUSES_PER_SIDE,
        "max_backtracks_per_cursor": MAX_BACKTRACKS_PER_CURSOR,
        "expansions": expansions,
        "prefix_matching_expansions": prefix_matches,
        "grammar_rejections": grammar_rejections,
        "completed_states": len(completed),
        "best_state": best,
        "obstruction": obstruction,
        "exact_closure": bool(completed),
    }


def seam_attempt(parent_rendered: str, parent_tape: str, windows: dict[str, tuple[int, int]]) -> dict[str, object]:
    left = parent_tape[windows["normalized_left"][0] : windows["normalized_left"][1]]
    right = parent_tape[windows["normalized_right"][0] : windows["normalized_right"][1]]
    left_context = boundary_context(parent_rendered, windows["raw_left"])
    right_context = boundary_context(parent_rendered, windows["raw_right"])
    parent_frames = frames_in_parent(parent_rendered)
    neighbor_map = {
        PRIMARY["normalized_left"]: ({"sara|saw", "leon|stops"}, {"leon|spots", "evil leon|was"}),
        SWITCH["normalized_left"]: ({"nora|sees", "sara|saw"}, {"leon|spots", "mara|sees"}),
    }
    left_neighbors, right_neighbors = neighbor_map[windows["normalized_left"]]
    left_neighbor_frames = parent_frames & left_neighbors
    right_neighbor_frames = parent_frames & right_neighbors
    left_result = expand_side("left", right[::-1], left_context, left_neighbor_frames)
    right_result = expand_side("right", left[::-1], right_context, right_neighbor_frames)
    closures = []
    for left_state in [left_result["best_state"]]:
        for right_state in [right_result["best_state"]]:
            if left_state["residual"] == right_state["residual"] == "":
                closures.append({"left": left_state, "right": right_state})
    return {
        "normalized_windows": {key: list(windows[key]) for key in ("normalized_left", "normalized_right")},
        "raw_windows": {key: list(windows[key]) for key in ("raw_left", "raw_right")},
        "required_reverse_prefix": {
            "left_obligation_derived_from_right": right[::-1],
            "right_obligation_derived_from_left": left[::-1],
        },
        "equation": {"left_length": len(left), "right_length": len(right), "right_is_reverse": right == left[::-1]},
        "neighboring_boundary_context": {"left": left_context, "right": right_context},
        "left_lattice": left_result,
        "right_lattice": right_result,
        "closures": closures,
        "exact_child_saved": False,
    }


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_independent = independent_audit(parent_rendered)
    assert parent_independent["normalized_letters"] == 666
    assert parent_independent["two_pointer_exact"]
    assert parent_independent["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)

    primary = seam_attempt(parent_rendered, parent_tape, PRIMARY)
    assert primary["equation"]["right_is_reverse"]
    assert not primary["closures"]
    switched = seam_attempt(parent_rendered, parent_tape, SWITCH)
    assert switched["equation"]["right_is_reverse"]
    assert not switched["closures"]

    row = {
        "id": "obligation-first-phrase-lattice-no-closure-666",
        "working_status": "obligation_lattice_rejected_no_exact_closure",
        "promotion_status": {
            "promoted": False,
            "status": "rejected_no_exact_child",
            "reason": "The obligation-first finite grammar exhausted the primary seam under bounded state and then switched once to a different actual seam; neither produced an independently exact changed child.",
        },
        "rendered": parent_rendered,
        "independent_audit": parent_independent,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "primary_attempt": primary,
        "switched_attempt": {
            "switch_reason": "immediate switch after primary seam obstruction",
            **switched,
        },
        "provenance": "obligation-first reciprocal prefix derivation, finite grammar phrase lattice, one bounded seam switch",
    }
    return {
        "experiment_id": "incumbent-666-obligation-first-phrase-lattice-20260922",
        "method": "derive required reverse prefix first; beam 16, max 128 expansions, four clauses per side, two backtracks per cursor",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": "continue from a different actual seam after the recorded switch obstruction; preserve current 666 and 568/560/558/556",
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    for label, attempt in (("primary", row["primary_attempt"]), ("switch", row["switched_attempt"])):
        print(label, {
            "left_expansions": attempt["left_lattice"]["expansions"],
            "right_expansions": attempt["right_lattice"]["expansions"],
            "left_cursor": attempt["left_lattice"]["obstruction"]["cursor"],
            "right_cursor": attempt["right_lattice"]["obstruction"]["cursor"],
            "left_reason": attempt["left_lattice"]["obstruction"]["reason"],
            "right_reason": attempt["right_lattice"]["obstruction"]["reason"],
        })


if __name__ == "__main__":
    main()
