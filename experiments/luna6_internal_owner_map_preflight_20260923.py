"""Record a bounded internal owner-map obstruction before lexical rendering."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUTPUT = ROOT / "runs/luna6-internal-owner-map-preflight-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"


def letters(text: str) -> str:
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def main() -> dict:
    payload = json.loads(PARENT.read_text())
    parent = next(r["rendered"] for r in payload["rows"]
                  if r["id"] == "outer-causal-scene-568-working-incumbent")
    tape = letters(parent)
    if len(tape) != 568 or sha(tape) != PARENT_SHA:
        raise AssertionError("Pinned parent identity changed")

    # A replacement of two mirrored four-letter owners can grow the parent
    # only if replacement widths balance; check one authored balanced pair.
    left = "Do not fear them"
    right = "They beg to nod"
    lt, rt = letters(left), letters(right)
    required = rt[::-1]
    cursor = 0
    while cursor < min(len(lt), len(required)) and lt[cursor] == required[cursor]:
        cursor += 1
    obstruction = {
        "normalized_parent_spans": {"left": [265, 269], "right": [299, 303]},
        "parent_owners": {"left": "Leon", "right": "Noel", "relation": "whole-token reversal; retired"},
        "replacement_clauses": {"left": left, "right": right},
        "normalized_widths": {"removed": [4, 4], "replacement": [len(lt), len(rt)],
                              "replacement_widths_balance": len(lt) == len(rt),
                              "nominal_total_growth": len(lt) + len(rt) - 8},
        "suffix_first_match": {
            "partner_final_sequence": "to nod",
            "right_final_sequence_tape": letters("to nod"),
            "reversed_prefix": required[:cursor],
            "left_opening": lt[:cursor],
            "matched_characters": cursor,
            "cursor": cursor,
            "left_char": lt[cursor],
            "required_char": required[cursor],
            "left_residual": lt[cursor:],
            "required_residual": required[cursor:]},
        "rendering_decision": "Do not render: the exact local equation blocks at the first content owner after the five-character cross-boundary match (`f/g`), and the 13/12 replacement widths do not balance. The nominal total growth is +17, but this owner map cannot close.",
    }

    # A separate earlier proposal is included only to explain why a tempting
    # expansion is retired; it is not a second realization.
    imbalanced = {"left": "Do not fret, folks", "right": "The guard was told to nod"}
    imbalanced_widths = [len(letters(imbalanced["left"])), len(letters(imbalanced["right"]))]
    return {
        "experiment_id": "luna6-internal-owner-map-preflight-20260923",
        "status": "preflight-obstruction-no-full-render",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "letters": 568,
                   "sha256": PARENT_SHA, "identity_unchanged": True},
        "balanced_owner_map_obstruction": obstruction,
        "retired_width_imbalance": {
            "proposal": imbalanced,
            "replacement_widths": imbalanced_widths,
            "left_minus_right": imbalanced_widths[0] - imbalanced_widths[1],
            "first_local_conflict": {"cursor": 5, "left": "f", "required": "d"},
            "decision": "retired before rendering: six-letter imbalance and early character conflict"},
        "next_operator": {
            "kind": "growth-producing active/passive role-carrying internal replacement",
            "different_parent_spans": {"left": [248, 265], "right": [303, 320]},
            "current_source_clauses": {"left": "Nadia delivers maps", "right": "Spam's reviled, Aidan"},
            "owner_plan": "At a different reflected sentence pair, transfer the active agent/theme of `Nadia delivers maps` into a passive clause with an overt recipient (`Aidan`) while a shared live role state owns the two surfaces; the opposite clause's terminal owner must be solved before filling the left predicate, not selected from a fixed ending bank.",
            "growth_gate": "sum of replacement normalized widths must exceed 34 letters",
            "novelty_gate": "preflight this exact reflected-span/role signature; do not reuse the suffix-first owner map or search partner endings",
            "stop_rule": "one paired owner realization; if first mismatch is forced, preserve its cursor and change construction operator rather than relexicalizing these clauses"},
        "reader_status": "No candidate rendered or admitted; no human readability claim."}


if __name__ == "__main__":
    result = main()
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"],
                      "cursor": result["balanced_owner_map_obstruction"]["suffix_first_match"]["cursor"],
                      "left_char": result["balanced_owner_map_obstruction"]["suffix_first_match"]["left_char"],
                      "required_char": result["balanced_owner_map_obstruction"]["suffix_first_match"]["required_char"],
                      "nominal_growth": result["balanced_owner_map_obstruction"]["normalized_widths"]["nominal_total_growth"],
                      "widths_balance": result["balanced_owner_map_obstruction"]["normalized_widths"]["replacement_widths_balance"]}, sort_keys=True))
