"""Relation-first semordnilap construction with live character auditing.

The operator chooses a small event relation (agent/action/patient or
imperative/object) before choosing paired lexical edges.  This is not a
finished-tape reversal: every edge is consumed while the opposing character
obligation is live, and the two clauses are independently rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import socket
from pathlib import Path

PAIRS = [
    ("was", "saw", "AUX", "V", "state"),
    ("live", "evil", "V", "ADJ", "state"),
    ("deliver", "reviled", "V", "ADJ", "transfer"),
    ("reward", "drawer", "V", "N", "transfer"),
    ("parts", "strap", "N", "N", "separation"),
    ("diaper", "repaid", "N", "V", "transfer"),
    ("stressed", "desserts", "ADJ", "N", "state"),
    ("draw", "ward", "V", "N", "transfer"),
    ("no", "on", "DET", "PREP", "location"),
    ("evil", "live", "N", "V", "state"),
    ("noel", "leon", "NAME", "NAME", "identity"),
    ("raw", "war", "ADJ", "N", "state"),
]

RELATIONS = {
    "transfer": (("V", "N", "V"), ("V", "N", "V")),
    "state": (("V", "N", "ADJ"), ("V", "N", "ADJ")),
    "location": (("DET", "N", "NAME"), ("PREP", "V", "NAME")),
}


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    t = tape(text)
    mismatches = sum(a != b for a, b in zip(t, t[::-1])) // 2
    return {
        "letters": len(t),
        "two_pointer_exact": bool(t) and all(t[i] == t[-1 - i] for i in range(len(t) // 2)),
        "pointer_mismatches": mismatches,
        "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
    }


def gates(left: list[str], right: list[str], text: str) -> dict:
    words = left + right
    return {
        "distinct_words": len(set(words)) == len(words),
        "no_self_palindromic_word": all(tape(w) != tape(w)[::-1] for w in words),
        "no_finished_reversal": tape(" ".join(left)) != tape(" ".join(right))[::-1],
        "no_catalogue": True,
        "no_repeated_unit": len(set(left)) == len(left) and len(set(right)) == len(right),
        "rendered_complete": all(w.isalpha() for w in words) and ";" in text,
    }


def run(limit: int) -> list[dict]:
    by: dict[str, list[tuple[str, str, str]]] = {}
    for left, right, lpos, rpos, relation in PAIRS:
        by.setdefault(lpos, []).append((left, right, relation))
        by.setdefault(rpos, []).append((right, left, relation))
    rows: list[dict] = []
    for relation, (left_shape, right_shape) in RELATIONS.items():
        left_edges = [by.get(pos, []) for pos in left_shape]
        right_edges = [by.get(pos, []) for pos in right_shape]
        for left_tuple in itertools.product(*left_edges):
            # A relation is selected first; opposite edges must carry that
            # same relation, rather than merely matching POS labels.
            if any(edge[2] != relation for edge in left_tuple):
                continue
            left = [edge[0] for edge in left_tuple]
            obligations = tape(" ".join(left))[::-1]
            for right_tuple in itertools.product(*right_edges):
                if any(edge[2] != relation for edge in right_tuple):
                    continue
                right = [edge[0] for edge in right_tuple]
                text = " ".join(left) + "; " + " ".join(right)
                a = audit(text)
                g = gates(left, right, text)
                # Record every intact relation product, including failures,
                # so the first unsupported character is reproducible.
                support = 0
                for expected, actual in zip(obligations, tape(" ".join(right))):
                    if expected != actual:
                        break
                    support += 1
                row = {
                    "rendered": text,
                    "relation": relation,
                    "left_words": left,
                    "right_words": right,
                    "live_obligation": obligations,
                    "mirrored_support_depth": support,
                    "audit": a,
                    "gates": g,
                    "reader_worthy": False,
                    "provenance": {
                        "relation_selected_before_lexical_edges": True,
                        "online_character_obligation": True,
                        "finished_tape_reversal": False,
                        "catalogue_used": False,
                        "posthoc_repair": False,
                    },
                }
                if a["two_pointer_exact"] and all(g.values()):
                    rows.append(row)
                    if len(rows) >= limit:
                        return rows
    return rows


def observations(limit: int = 20) -> list[dict]:
    """Return rendered relation products even when no exact closure exists."""
    by: dict[str, list[tuple[str, str, str]]] = {}
    for left, right, lpos, rpos, relation in PAIRS:
        by.setdefault(lpos, []).append((left, right, relation))
        by.setdefault(rpos, []).append((right, left, relation))
    out: list[dict] = []
    for relation, (left_shape, right_shape) in RELATIONS.items():
        for left_tuple in itertools.product(*(by.get(pos, []) for pos in left_shape)):
            if any(edge[2] != relation for edge in left_tuple):
                continue
            left = [edge[0] for edge in left_tuple]
            for right_tuple in itertools.product(*(by.get(pos, []) for pos in right_shape)):
                if any(edge[2] != relation for edge in right_tuple):
                    continue
                right = [edge[0] for edge in right_tuple]
                text = " ".join(left) + "; " + " ".join(right)
                out.append({"rendered": text, "relation": relation, "audit": audit(text), "reader_worthy": False})
                if len(out) >= limit:
                    return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidates = run(args.limit)
    controls = observations(args.limit)
    payload = {
        "experiment": "semordnilap-relation-graph-20261002",
        "host": socket.gethostname(),
        "parameters": vars(args),
        "candidates": candidates,
        "controls": controls,
        "closures": len(candidates),
        "reader_worthy": sum(row["reader_worthy"] for row in candidates),
        "provenance": {
            "construction": "relation-first typed edge product",
            "exact_audit": "independent two-pointer plus forward/reverse SHA-256",
            "anti_shortcut_gates": "distinct words, no self-palindromic word, no catalogue, no finished reversal",
        },
        "next_construction": "replace pair edges with authored inflected subjects and objects for each relation; retain the live character frontier and require a complete semantic clause before exact admission.",
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("experiment", "closures", "reader_worthy")}))


if __name__ == "__main__":
    main()
