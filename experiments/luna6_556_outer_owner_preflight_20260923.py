"""Record the bounded novelty preflight for the preserved 556 frontier."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = ROOT / "runs/incumbent-498-event-frame-seam-repair-20261002.json"
OUT = ROOT / "runs/luna6-556-outer-owner-preflight-20260923.json"
ROW_ID = "depth39-longest-f1g1h1r"
EXPECTED_SHA = "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def token_spans(text: str) -> list[dict]:
    rows = []
    for m in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        start = len(letters(text[:m.start()]))
        end = len(letters(text[:m.end()]))
        rows.append({"token": m.group(), "span": [start, end]})
    return rows


def run() -> dict:
    payload = json.loads(PARENT.read_text())
    row = next(r for r in payload["rows"] if r["id"] == ROW_ID)
    rendered = row["rendered"]
    tape = letters(rendered)
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 556 or digest != EXPECTED_SHA:
        raise AssertionError("Pinned 556 frontier identity changed")
    if tape != tape[::-1]:
        raise AssertionError("Pinned 556 frontier is no longer exact")

    from llm_palindrome.validator import is_palindrome

    if not is_palindrome(rendered):
        raise AssertionError("Project validator rejects pinned frontier")

    history = [
        {
            "experiment": "experiments/incumbent_498_event_frame_seam_repair_20261002.py",
            "artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json",
            "signature": "creates the exact 556 row with the outer Nadia/delivers/maps … Spam's/reviled/Aidan shell and records the depth-39 i|ts lineage",
        },
        {
            "experiment": "experiments/incumbent_498_live_seam_growth_20261002.py",
            "artifact": "runs/incumbent-498-live-seam-growth-20261002.json",
            "signature": "explicit incumbent-specific partial-word owner/residual/cursor growth on this 498→556 lineage; does not establish a new outer owner map",
        },
        {
            "experiment": "experiments/incumbent_568_repeated_shell_event_lattice_20261002.py",
            "artifact": "runs/incumbent-568-repeated-shell-event-lattice-20261002.json",
            "signature": "repeated outer event-shell resegmentation on the same 556 row; the Nadia/Spam shell is already the repeated-shell region",
        },
        {
            "experiment": "experiments/incumbent_568_live_partial_seam_growth_20261002.py",
            "artifact": "runs/incumbent-568-live-partial-seam-growth-20261002.json",
            "signature": "partial-seam growth lineage on the 556 parent; central i|ts closure and related partial-seam sweeps are excluded by the task",
        },
    ]
    for item in history:
        for key in ("experiment", "artifact"):
            if not (ROOT / item[key]).exists():
                raise AssertionError(f"Missing history source: {item[key]}")

    spans = token_spans(rendered)
    by_span = {tuple(t["span"]): t["token"] for t in spans}
    owner_audit = [
        {
            "left_span": [5, 13],
            "left_owner": "delivers",
            "reflected_interval": [543, 551],
            "reflected_source_owners": [
                {"span": [539, 544], "token": "Spam's", "contribution": "s"},
                {"span": [544, 551], "token": "reviled", "contribution": "reviled"},
            ],
            "equation": "reverse(delivers) = s + reverse(reviled)",
            "novelty": "collision: this is the existing outer event shell in the 556 parent and its frame-repair/repeated-shell lineage",
        },
        {
            "left_span": [13, 17],
            "left_owner": "maps",
            "reflected_interval": [539, 543],
            "reflected_source_owners": [
                {"span": [539, 544], "token": "Spam's", "contribution": "spam"}
            ],
            "equation": "reverse(maps) = first four letters of Spam's",
            "novelty": "collision: same existing Nadia/Spam outer shell; not an independent owner map",
        },
        {
            "left_span": [63, 71],
            "left_owner": "a post its",
            "reflected_interval": [485, 493],
            "reflected_source_owners": [
                {"span": [483, 487], "token": "past", "contribution": "ts"},
                {"span": [487, 490], "token": "its", "contribution": "i"},
                {"span": [490, 493], "token": "Opa", "contribution": "opa"},
            ],
            "equation": "the depth-39 i|ts / post|its seam family",
            "novelty": "excluded: this is the central owner map the task expressly forbids rerunning",
        },
    ]
    # Keep the token map used to verify the hard-coded owner spans visible.
    for item in owner_audit:
        for owner in item["reflected_source_owners"]:
            if tuple(owner["span"]) not in by_span:
                # `contribution` may be only a subspan of a token, but the
                # declared source owner itself must exist in the token map.
                raise AssertionError(f"Reflected owner span absent: {owner}")

    return {
        "experiment_id": "luna6-556-outer-owner-preflight-20260923",
        "status": "no_candidate_attempted_no_untried_outer_partial_word_map_justified",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "row": ROW_ID,
            "rendered": rendered,
            "normalized_letters": len(tape),
            "normalized_sha256": digest,
            "independent_outside_in_exact": tape == tape[::-1],
            "project_validator_exact": bool(is_palindrome(rendered)),
        },
        "scope": "One bounded novelty preflight on the preserved 556 frontier; no candidate drafting, no central i|ts attempt, no generic insertion or repeated-shell sweep.",
        "outer_owner_map": owner_audit[:2],
        "excluded_central_owner_map": owner_audit[2],
        "history_collisions": history,
        "decision": "The two outer partial equations are subowners of the same already-used repeated event shell. The next partial seam belongs to the forbidden depth-39 i|ts lineage. Thus no untried outer partial-word owner map with a new grammatical variable is evidenced on this parent.",
        "next_action": "Return to coordinator for a non-overlapping method on the pinned 568 incumbent; this artifact does not launch or reserve a method already assigned to another lane.",
        "reader_status": "No new candidate was produced; no readability claim.",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run": str(OUT), "status": result["status"],
                      "owner_maps": len(result["outer_owner_map"]),
                      "exact_parent": result["parent"]["independent_outside_in_exact"] and result["parent"]["project_validator_exact"]}, indent=2))
