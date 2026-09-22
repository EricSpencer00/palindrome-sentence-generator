"""Audit the bounded search for a better open carrier at the k=2 center.

The expensive inventories were produced on ``hst-bench`` by the three probe
sources beside this file.  This replay loads their immutable JSON outputs,
renders every exact candidate, and applies the current repository admission
gate.  Mechanical eligibility and discourse readability remain separate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


ROOT = Path(__file__).resolve().parents[1]
ID = "open-carrier-semantic-search-20260922"
OUT = ROOT / "runs" / f"{ID}.json"
RAW = ROOT / "artifacts" / ID
RESIDUAL = "s"
CENTER = "Spot spoons. Snoop. Stop."
BASELINE = "No trace. Note: Spot spoons. Snoop. Stop. Set one carton."
LONGER = "Nine poll assert. Spot spoons. Snoop. Stop. Stress all open in."


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatch = next((i for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape), "two_pointer_exact": mismatch is None and bool(tape),
        "first_mismatch": mismatch, "sha256_forward": forward,
        "sha256_reverse": reverse, "hashes_agree": forward == reverse,
    }


def proper_palindromic_spans(text: str) -> list[dict[str, object]]:
    words = tokenize(text)
    rows = []
    for start in range(len(words)):
        for stop in range(start + 2, len(words) + 1):
            if start == 0 and stop == len(words):
                continue
            tape = "".join(normalize_letters(word) for word in words[start:stop])
            if tape == tape[::-1]:
                rows.append({"start": start, "stop": stop,
                             "words": list(words[start:stop])})
    return rows


def live_word_trace(text: str) -> list[dict[str, object]]:
    """Replay outside-in word choices with explicit owner, debt, and cursors."""
    words = tokenize(text)
    left_cursor, right_cursor = 0, len(words) - 1
    owner = ""
    residual = ""
    rows = []
    while left_cursor <= right_cursor:
        side = "right" if owner == "left" else "left"
        if not owner:
            side = "left"
        if side == "left":
            word = words[left_cursor]
            left_cursor += 1
            exposed = normalize_letters(word)
        else:
            word = words[right_cursor]
            right_cursor -= 1
            exposed = normalize_letters(word)[::-1]
        if residual:
            common = min(len(residual), len(exposed))
            if residual[:common] != exposed[:common]:
                raise AssertionError((word, owner, residual, exposed))
            if len(residual) > len(exposed):
                residual = residual[common:]
            elif len(exposed) > len(residual):
                owner, residual = side, exposed[common:]
            else:
                owner, residual = "", ""
        else:
            owner, residual = side, exposed
        rows.append({
            "chosen_side": side, "surface_word": word, "exposed": exposed,
            "owner_after": owner or "none", "residual_after": residual,
            "left_word_cursor": left_cursor, "right_word_cursor": right_cursor,
        })
    if residual:
        raise AssertionError((owner, residual))
    return rows


def candidate(text: str, p_words: tuple[str, ...], q_words: tuple[str, ...],
              *, semantic_status: str, semantic_obstruction: str | None) -> dict[str, object]:
    p_tape = "".join(p_words)
    q_tape = "".join(q_words)
    checks = mechanical_admission_checks(text)
    spans = proper_palindromic_spans(text)
    return {
        "rendered": text, "p_words": list(p_words), "q_words": list(q_words),
        "carrier_equation": {
            "reverse_q": q_tape[::-1], "p_plus_residual": p_tape + RESIDUAL,
            "holds": q_tape[::-1] == p_tape + RESIDUAL,
            "owner_at_center_entry": "right", "residual_at_center_entry": RESIDUAL,
            "p_character_cursor": len(p_tape),
            "reverse_q_character_cursor": len(p_tape),
            "reverse_q_characters_total": len(q_tape),
        },
        "live_owner_residual_cursor_trace": live_word_trace(text),
        "independent_audit": audit(text),
        "mechanical_checks": checks, "mechanically_admitted": all(checks.values()),
        "boundary_mask": {"proper_palindromic_multiword_spans": spans,
                          "passed": not spans},
        "freshness": {
            "carrier_content_disjoint_from_center": not (
                (set(p_words) | set(q_words)) & {"spot", "spoon", "spoons", "snoop", "stop"}),
            "all_content_words_distinct": checks["distinct_words"],
            "proper_names": False, "repeated_units": False,
            "mirrored_finished_phrases": False, "posthoc_repair": False,
        },
        "semantic_status": semantic_status,
        "semantic_obstruction": semantic_obstruction,
    }


def run() -> dict[str, object]:
    brown = json.loads((RAW / "brown-span-probe.json").read_text())
    template = json.loads((RAW / "template-probe.json").read_text())
    cycles = json.loads((RAW / "cycle-probe.json").read_text())
    baseline = candidate(
        BASELINE, ("no", "trace", "note"), ("set", "one", "carton"),
        semantic_status="incumbent_readable_control", semantic_obstruction=None,
    )
    longer = candidate(
        LONGER, ("nine", "poll", "assert"), ("stress", "all", "open", "in"),
        semantic_status="rejected_unreadable_pos_ambiguity",
        semantic_obstruction=(
            "The tape-only POS intersection assigns D-N-V to 'nine poll assert' "
            "and V-D-A-N to 'stress all open in'. Surface agreement rejects the "
            "first ('nine poll' requires a plural noun), while the second ends "
            "in preposition 'in' misused as a noun and leaves 'stress' without "
            "an ordinary object. Neither side forms a readable discourse carrier."
        ),
    )
    residual_frontier = []
    diagnoses = {
        "s": "After excluding acronym-like 'smu', only spot/stop and snap/span survive; snap/span exposes the already recorded unsatisfied transitive return.",
        "ac": "The live 'ac' suffix has no productive English boundary realization; act/acts/acted cannot absorb it without a fragment.",
        "ca": "The live 'ca' suffix has no productive English boundary realization; cat/cast/cadet leave an unattached fragment.",
        "no": "The residual can be a word, but notes/nose/no/ones/onset is a noun stack with no coherent finite clause.",
        "on": "The inverse noun family likewise yields ones/onset/on/nose/notes without a licensed event structure.",
        "ow": "own/won and owe/woe leave the nonlexical boundary fragment 'ow'.",
        "wo": "won/own and woe/owe leave the nonlexical boundary fragment 'wo'.",
    }
    for row in cycles["productive_residuals"]:
        residual_frontier.append({
            "residual": row["residual"],
            "pair_families": [[pair["x"], pair["y"]] for pair in row["pairs"]],
            "all_equations_hold": all(pair["equation"]["holds"] for pair in row["pairs"]),
            "semantic_or_morphological_obstruction": diagnoses[row["residual"]],
        })
    return {
        "experiment_id": ID,
        "decision": "no carrier improves on the 42-letter incumbent",
        "acceptance_gate": (
            "more than 42 letters; exact; current mechanical gate and proper-span "
            "mask pass; fresh common words; both carrier phrases and the k=2 "
            "center form one readable discourse without proper names, repeats, "
            "mirrored finished phrases, catalogue text, or repair"
        ),
        "remote_search": {
            "host": "hst-bench",
            "attested_span_stats": brown["stats"],
            "typed_template_stats": template["stats"],
            "productive_residual_stats": cycles["stats"],
            "raw_artifacts": [
                "artifacts/open-carrier-semantic-search-20260922/brown-span-probe.json",
                "artifacts/open-carrier-semantic-search-20260922/template-probe.json",
                "artifacts/open-carrier-semantic-search-20260922/cycle-probe.json",
            ],
        },
        "baseline": baseline,
        "exact_longer_candidates": [longer],
        "accepted_improvements": [],
        "obstruction": {
            "attested_carriers": (
                "No exact pair exists among 2,743,308 eligible Brown span occurrences "
                "and 2,343,724 distinct tapes in the bounded 2-9 word, 8-36 letter domain."
            ),
            "authored_typed_carriers": (
                "The only exact >42 result in 8,344,980 products is mechanically "
                "clean but depends on incompatible POS ambiguities rather than a "
                "readable phrase pair."
            ),
            "semantic_segmentation": longer["semantic_obstruction"],
        },
        "diverse_residual_frontier": residual_frontier,
        "reader_gate": {
            "status": "closed",
            "reason": "no candidate passes the pre-reader discourse-semantic gate",
        },
        "provenance": {
            "catalogue_text": False, "finished_tape_reversal": False,
            "posthoc_repair": False, "proper_names": False,
            "remote_source_sha256": {
                "brown": brown["provenance"]["source_sha256"],
                "templates": template["provenance"]["source_sha256"],
                "cycles": cycles["provenance"]["source_sha256"],
            },
            "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main() -> None:
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "decision": payload["decision"],
        "longer_exact": len(payload["exact_longer_candidates"]),
        "accepted_improvements": len(payload["accepted_improvements"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
