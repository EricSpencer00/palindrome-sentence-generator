"""Repair dead residuals with typed lexical substitutions, then resume trie search."""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.boundary_residual_typed_trie_20260914 import build_trie
from experiments.syntax_first_clause_pair_20260914 import (
    exact_audit, enumerate_clauses, normalize,
)

# Fresh authored alternatives, partitioned by syntactic role.  They are not
# scored or borrowed from a palindrome list; substitutions preserve the role.
REPAIRS = {
    "subject": ("author", "doctor", "farmer", "judge", "nurse", "sailor"),
    "verb": ("builds", "checks", "cuts", "draws", "guides", "marks", "plans", "writes"),
    "object": ("answer", "chart", "draft", "garden", "image", "lesson", "story", "task"),
    "adj": ("bright", "gentle", "honest", "plain", "ready", "warm"),
    "place": ("arena", "harbor", "island", "market", "station", "valley"),
}


def repaired_rows(rows: list[dict], limit: int) -> list[dict]:
    out = []
    seen = set()
    for row in rows[:limit]:
        roles = row["roles"]
        for index, role in enumerate(roles):
            if role not in REPAIRS:
                continue
            for replacement in REPAIRS[role]:
                words = list(row["words"])
                words[index] = replacement
                text = " ".join(words)
                tape = normalize(text)
                key = (tape, tuple(words), row["plan"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({**row, "text": text, "words": tuple(words), "tape": tape,
                            "repair": {"role": role, "slot": index,
                                       "replacement": replacement}})
    return out


def run(max_base: int = 60_000, max_repairs: int = 180_000, max_pairs: int = 100) -> dict:
    base = enumerate_clauses()
    left = base[:max_base]
    repairs = repaired_rows(base, max_repairs)
    right_rows = base + repairs
    trie = build_trie(right_rows)
    closures = []
    dead_before = dead_after = 0
    max_consumed = 0
    for row in left + repairs:
        node = 0
        consumed = 0
        for char in row["tape"]:
            nxt = trie[node].children.get(char)
            if nxt is None:
                break
            node, consumed = nxt, consumed + 1
        max_consumed = max(max_consumed, consumed)
        if consumed == 0:
            dead_before += 1
        if consumed < len(row["tape"]):
            dead_after += 1
            continue
        for idx in trie[node].leaves:
            other = right_rows[idx]
            # Avoid reporting the same surface pair twice and require at
            # least one actual repair in the reported construction.
            if not (row.get("repair") or other.get("repair")):
                continue
            rendered = row["text"].capitalize() + "; " + other["text"] + "."
            audit = exact_audit(rendered)
            if not audit["exact"]:
                raise AssertionError("repair trie emitted a non-palindrome")
            if len(closures) < max_pairs:
                closures.append({"left": row, "right": other, "rendered": rendered,
                                 "independent_exact_audit": audit,
                                 "reader_status": "human-unreviewed"})
    return {
        "status": "role_preserving_boundary_repair",
        "config": {"max_base_left": max_base, "max_repair_source": max_repairs,
                    "max_reported_pairs": max_pairs, "resume_same_trie": True,
                    "role_preserving": True, "complete_clause_reparse": True,
                    "catalogue_text": False, "known_palindrome_units": False},
        "inventory": {"base_clauses": len(base), "repaired_clauses": len(repairs),
                       "trie_rows": len(right_rows), "trie_nodes": len(trie),
                       "searched_left_rows": len(left) + len(repairs)},
        "residual_evidence": {"dead_at_first_character": dead_before,
                              "dead_before_full_tape": dead_after,
                              "maximum_characters_consumed": max_consumed},
        "exact_closure_count_seen": len(closures), "rendered_candidates": closures,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "repairs": "fresh authored role-specific lexical alternatives in this file"},
        "next_constructive_operator": "Permit two coordinated role repairs at the same residual boundary, with number/valency features checked after each complete clause; retain only novel exact closures for human review.",
        "scope": "Bounded repair search; exactness is independently checked and readability remains a blinded-human question.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-base", type=int, default=60_000)
    ap.add_argument("--max-repairs", type=int, default=180_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    result = run(args.max_base, args.max_repairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"base": result["inventory"]["base_clauses"],
                      "repairs": result["inventory"]["repaired_clauses"],
                      "closures": result["exact_closure_count_seen"]}, indent=2))


if __name__ == "__main__":
    main()
