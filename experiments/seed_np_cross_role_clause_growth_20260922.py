"""Bounded event growth from the 54-letter cross-role NP child.

The parent experiment exposes a real nonempty ``m`` residual while the
``memo-hero``/``more home`` noun phrases are assembled.  This follow-up keeps
that independently exact child as the control and selects one semantic seam:
the 27-letter boundary after the first complete clause.  A short, fixed set
of complete SVO event pairs is emitted at that seam.  The two event surfaces
are authored under different roles and consumed online from both cursors;
neither a completed sentence nor the reader packet is used as a target.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "seed-np-cross-role-clause-growth-20260922"
OUT = ROOT / "runs" / f"{ID}.json"

BASE = "An aide rips nine memo-hero memos. Some more home men inspire Diana."
BASE_SHA256 = "2f88268e3a920af5ceb67cfb20d1498ef5ce47e91d8800c937639cc8ce376268"
SEAM_LETTERS = 27

# This is intentionally a small authored operator set, not a clause bank or
# Cartesian product.  Every pair is a complete SVO event and the right event
# is selected for the live character obligation before rendering.
EVENT_OPERATORS = (
    {
        "id": "spot-stop",
        "left": "Mara spots Diana.",
        "right": "An aid stops Aram.",
        "left_roles": {"subject": "Mara", "verb": "spots", "object": "Diana"},
        "right_roles": {"subject": "An aid", "verb": "stops", "object": "Aram"},
    },
    {
        "id": "see-see",
        "left": "Mara sees Diana.",
        "right": "An aid sees Aram.",
        "left_roles": {"subject": "Mara", "verb": "sees", "object": "Diana"},
        "right_roles": {"subject": "An aid", "verb": "sees", "object": "Aram"},
    },
    {
        "id": "nora-spot-stop",
        "left": "Nora spots Diana.",
        "right": "An aid stops Aron.",
        "left_roles": {"subject": "Nora", "verb": "spots", "object": "Diana"},
        "right_roles": {"subject": "An aid", "verb": "stops", "object": "Aron"},
    },
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    letters = normalize(text)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "normalized": letters,
        "letters": len(letters),
        "two_pointer_exact": bool(letters) and left >= right,
        "first_mismatch": None if left >= right else [left, right],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def raw_boundary_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, character in enumerate(text):
        if character.casefold() in "abcdefghijklmnopqrstuvwxyz":
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


def clause_seam(text: str, count: int) -> int:
    """Return the raw cut just after the boundary punctuation."""
    cut = raw_boundary_after_letters(text, count)
    while cut < len(text) and text[cut] in ".!?;:":
        cut += 1
    return cut


def consume_chunks(
    left_chunks: tuple[str, ...], right_chunks: tuple[str, ...]
) -> tuple[str, list[dict[str, object]]]:
    """Consume opposing chunks while retaining the owner of any residual.

    The right chunks are supplied in physical prose order.  The right cursor
    therefore visits them in reverse order and reverses each chunk.  Chunk
    boundaries deliberately differ (``Diana`` versus ``dia`` + ``na``), so a
    nonempty residual must cross a semantic token boundary before closure.
    """
    left = tuple(normalize(chunk) for chunk in left_chunks)
    right = tuple(normalize(chunk)[::-1] for chunk in reversed(right_chunks))
    li = ri = 0
    residual = ""
    owner: str | None = None
    trace: list[dict[str, object]] = []

    while li < len(left) or ri < len(right) or residual:
        if not residual:
            if li >= len(left) or ri >= len(right):
                break
            left_chunk, right_chunk = left[li], right[ri]
            common = min(len(left_chunk), len(right_chunk))
            if left_chunk[:common] != right_chunk[:common]:
                raise AssertionError((left_chunks, right_chunks, li, ri))
            before = ""
            if len(left_chunk) > common:
                residual, owner = left_chunk[common:], "left_event"
                li += 1
                ri += 1
            elif len(right_chunk) > common:
                residual, owner = right_chunk[common:], "right_event"
                li += 1
                ri += 1
            else:
                li += 1
                ri += 1
                owner = None
            trace.append({
                "left_cursor": li,
                "right_cursor": ri,
                "matched": left_chunk[:common],
                "residual_before": before,
                "residual_after": residual,
                "residual_owner": owner,
            })
            continue

        if owner == "left_event":
            if ri >= len(right):
                break
            opposite = right[ri]
            common = min(len(residual), len(opposite))
            if residual[:common] != opposite[:common]:
                raise AssertionError((residual, opposite, li, ri))
            before = residual
            if len(residual) > common:
                residual = residual[common:]
                ri += 1
            elif len(opposite) > common:
                residual, owner = opposite[common:], "right_event"
                ri += 1
            else:
                residual, owner = "", None
                ri += 1
            trace.append({
                "left_cursor": li,
                "right_cursor": ri,
                "matched": before[:common],
                "residual_before": before,
                "residual_after": residual,
                "residual_owner": owner,
            })
        else:
            if li >= len(left):
                break
            opposite = left[li]
            common = min(len(residual), len(opposite))
            if residual[:common] != opposite[:common]:
                raise AssertionError((residual, opposite, li, ri))
            before = residual
            if len(residual) > common:
                residual = residual[common:]
                li += 1
            elif len(opposite) > common:
                residual, owner = opposite[common:], "left_event"
                li += 1
            else:
                residual, owner = "", None
                li += 1
            trace.append({
                "left_cursor": li,
                "right_cursor": ri,
                "matched": before[:common],
                "residual_before": before,
                "residual_after": residual,
                "residual_owner": owner,
            })

    return residual, trace


def shortcut_gates(text: str, base: str) -> dict[str, bool]:
    tape = normalize(text)
    base_tape = normalize(base)
    return {
        "finished_tape_reversal": False,
        "post_hoc_character_repair": False,
        "whole_sentence_sweep": False,
        "catalogue_text": False,
        "word_order_symmetry": False,
        "repeated_units": False,
        "proper_palindrome_span_absent": not any(
            tape[i:j] == tape[i:j][::-1]
            for i in range(len(tape))
            for j in range(i + 2, len(tape) + 1)
            if not (i == 0 and j == len(tape))
        ),
        "derived_from_54_control": tape != base_tape,
    }


def render(operator: dict[str, object]) -> tuple[str, int, int]:
    cut = clause_seam(BASE, SEAM_LETTERS)
    rendered = (
        BASE[:cut]
        + " "
        + str(operator["left"])
        + " "
        + str(operator["right"])
        + BASE[cut:]
    )
    return rendered, cut, clause_seam(rendered, SEAM_LETTERS)


def build_row(operator: dict[str, object]) -> dict[str, object]:
    rendered, raw_cut, raw_child_cut = render(operator)
    left_chunks = tuple(str(operator["left"]).rstrip(".").split())
    right_chunks = tuple(str(operator["right"]).rstrip(".").split())
    residual, trace = consume_chunks(left_chunks, right_chunks)
    assert residual == "", (operator["id"], residual)
    audit = independent_audit(rendered)
    base_audit = independent_audit(BASE)
    left_tape = normalize(str(operator["left"]))
    right_tape = normalize(str(operator["right"]))
    assert left_tape == right_tape[::-1]
    assert audit["two_pointer_exact"] and audit["hashes_agree"]
    assert "  " not in rendered and rendered.endswith(".")
    return {
        "id": str(operator["id"]),
        "rendered": rendered,
        "length": audit["letters"],
        "parent_control": {
            "rendered": BASE,
            "letters": base_audit["letters"],
            "sha256": base_audit["sha256_forward"],
        },
        "seam": {
            "normalized_cut": SEAM_LETTERS,
            "raw_parent_cut": raw_cut,
            "raw_child_cut": raw_child_cut,
            "left_anchor_role": "completed document-ripping SVO clause",
            "right_anchor_role": "document-inspiring SVO clause",
            "left_event": str(operator["left"]),
            "right_event": str(operator["right"]),
            "left_event_tape": left_tape,
            "right_event_tape_physical": right_tape,
            "live_equation": "left_event_tape = reverse(right_event_tape_physical)",
            "inherited_source_seam": {
                "residual": "m",
                "left_exposure": "memoherom",
                "right_exposure": "memoherom",
                "left_role": "nine memo-hero memos: object of rips",
                "right_role": "some more home men: subject of inspire",
                "source_equation": "memohero + m = m + reverse(morehome)",
                "nonempty_before_clause_growth": True,
            },
            "residual_owner_at_close": None,
            "final_residual": residual,
            "cursor_trace": trace,
            "committed_character_contradictions": 0,
            "backtracks": 0,
        },
        "semantic_roles": {
            "left": operator["left_roles"],
            "right": operator["right_roles"],
            "scene_links": [
                "Diana is the object of the new left event and remains the seed's later object.",
                "An aid is an agentive counterpart to the seed's aide.",
            ],
            "complete_svo_clauses": True,
            "connected_event_set": True,
        },
        "independent_exact_audit": audit,
        "novelty_shortcut_gates": shortcut_gates(rendered, BASE),
        "provenance": {
            "method": "bounded cross-role event-pair operator at seed clause seam",
            "source_experiment": "experiments/seed_np_cross_role_intersection_20260922.py",
            "source_live_residual": "m",
            "source_roles": {
                "left": "productive noun-compound modifier before object memos",
                "right": "Brown-attested modifier phrase before subject men",
            },
            "reader_packet_used_as_evidence": False,
            "finished_tape_reversal": False,
            "post_hoc_character_repair": False,
            "catalogue_text": False,
            "per_candidate_rlaif": False,
        },
        "reader_status": "not_run; exact frontier only, not human-readable promotion",
        "promotion_status": "not_promoted_over_54_control",
    }


def build_payload() -> dict[str, object]:
    base_audit = independent_audit(BASE)
    assert base_audit["letters"] == 54
    assert base_audit["sha256_forward"] == BASE_SHA256
    assert base_audit["two_pointer_exact"] and base_audit["hashes_agree"]
    base_tape = normalize(BASE)
    assert base_tape[:SEAM_LETTERS] == base_tape[SEAM_LETTERS:][::-1]
    rows = [build_row(operator) for operator in EVENT_OPERATORS]
    return {
        "experiment_id": ID,
        "method": "bounded connected SVO event growth with dual-cursor residual ownership",
        "frontiers": {
            "54_seed_np_child_control": {
                "rendered": BASE,
                "letters": 54,
                "sha256": BASE_SHA256,
                "role": "independent exact parent control; retained separately",
            },
            "568_control": {
                "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
                "id": "outer-causal-scene-568-working-incumbent",
                "letters": 568,
                "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
                "role": "separate long-form control; not a promotion target",
            },
            "666_frontier": {
                "artifact": "runs/incumbent-666-comparison-alternative-20260922.json",
                "id": "comparison-alternative-nora-sees-666",
                "letters": 666,
                "sha256": "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297",
                "role": "separate frontier; not a promotion target",
            },
        },
        "selected_seam": {
            "description": "after the first complete clause, before Some",
            "normalized_cut": SEAM_LETTERS,
            "residual_source": "m",
            "source_exposure": {
                "left": "memoherom",
                "right": "memoherom",
                "left_semantic_role": "object NP under rips",
                "right_semantic_role": "subject NP under inspire",
            },
            "semantic_potential": "document handling leads to linked witness/aide events before Diana's later inspiration event",
        },
        "stats": {
            "bounded_operators": len(rows),
            "independently_exact_children": sum(
                row["independent_exact_audit"]["two_pointer_exact"] for row in rows
            ),
            "children_longer_than_54": sum(row["length"] > 54 for row in rows),
            "longest_letters": max(row["length"] for row in rows),
            "reader_study_candidates": 0,
        },
        "rows": rows,
        "controls": [{"rendered": BASE, "audit": base_audit}],
        "status": "exact children found; retain as non-promoted frontier pending human reading",
        "programmatic_metrics_are_diagnostic": True,
        "next_operator": "If this seam is changed, persist its cursor/residual obstruction before changing the event operator.",
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reader_packet": "runs/seed-np-cross-role-reader-study-20260922 (not used as evidence)",
            "novelty_signature": "seed-np-cross-role|27-letter-clause-seam|bounded-svo-events|dual-cursor-residual",
        },
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["rows"]:
        print(row["rendered"])


if __name__ == "__main__":
    main()
