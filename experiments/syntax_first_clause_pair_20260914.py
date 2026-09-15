"""Syntax-first clause-pair matching for exact letter palindromes.

This branch enumerates authored, typed intact clauses before doing any tape
matching.  A clause is retained only if its independent reverse-tape partner
is also a complete typed clause.  The inventories are deliberately local and
ordinary; no catalogue strings or known-palindrome units are loaded.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z]+")

DETS = ("a", "an", "the")
SUBJECTS = ("artist", "captain", "editor", "teacher", "writer", "pilot", "agent")
VERBS = ("drafts", "draws", "finds", "holds", "keeps", "leads", "makes", "opens", "reads", "sends", "sets", "sees")
OBJECTS = ("book", "door", "gift", "letter", "map", "note", "plan", "report", "road", "room")
ADJS = ("brief", "calm", "clear", "kind", "new", "old", "open", "quiet", "red", "small")
ADVS = ("carefully", "early", "gently", "often", "quietly", "slowly", "well")
PREPS = ("by", "in", "near", "on", "with")
PLACES = ("garden", "office", "park", "studio", "town")


def normalize(text: str) -> str:
    return "".join(WORD.findall(text.lower()))


def exact_audit(text: str) -> dict:
    tape = normalize(text)
    mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2)
                  if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def plans() -> tuple[dict, ...]:
    # Slot order is syntax, not a post-hoc score.  Each clause is a complete
    # proposition with explicit subject, transitive verb, and object.
    return (
        {"name": "svo", "roles": ("det", "subject", "verb", "det", "object"),
         "choices": (DETS, SUBJECTS, VERBS, DETS, OBJECTS)},
        {"name": "adj_svo", "roles": ("det", "adj", "subject", "verb", "det", "adj", "object"),
         "choices": (DETS, ADJS, SUBJECTS, VERBS, DETS, ADJS, OBJECTS)},
        {"name": "adv_svo", "roles": ("det", "subject", "verb", "adv", "det", "object"),
         "choices": (DETS, SUBJECTS, VERBS, ADVS, DETS, OBJECTS)},
        {"name": "pp_svo", "roles": ("det", "subject", "verb", "det", "object", "prep", "det", "place"),
         "choices": (DETS, SUBJECTS, VERBS, DETS, OBJECTS, PREPS, DETS, PLACES)},
    )


def enumerate_clauses() -> list[dict]:
    rows = []
    for plan in plans():
        for words in itertools.product(*plan["choices"]):
            text = " ".join(words)
            rows.append({"text": text, "words": words, "plan": plan["name"],
                         "roles": plan["roles"], "tape": normalize(text)})
    return rows


def run() -> dict:
    clauses = enumerate_clauses()
    by_tape = {row["tape"]: row for row in clauses}
    pairs = []
    for row in clauses:
        partner = by_tape.get(row["tape"][::-1])
        if partner is None or row["tape"] > partner["tape"]:
            continue
        # The pair forms one intact two-clause string.  Independent audit is
        # intentionally separate from the construction lookup.
        rendered = row["text"].capitalize() + "; " + partner["text"] + "."
        pairs.append({"left": row, "right": partner, "rendered": rendered,
                      "independent_exact_audit": exact_audit(rendered),
                      "reader_status": "human-unreviewed"})
    return {
        "status": "syntax_first_clause_pair_matching",
        "config": {"typed_clause_enumeration_before_tape_matching": True,
                    "intact_propositions_only": True, "catalogue_text": False,
                    "known_palindrome_units": False, "independent_exact_audit": True},
        "inventory": {"plans": [{"name": p["name"], "roles": list(p["roles"]),
                                  "sizes": [len(c) for c in p["choices"]]} for p in plans()]},
        "enumerated_clause_count": len(clauses), "pairs": pairs,
        "exact_pair_count": sum(x["independent_exact_audit"]["exact"] for x in pairs),
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "authored ordinary lexical pools in this file",
                       "construction": "complete typed clause enumeration then reversed tape lookup"},
        "next_constructive_operator": "Use the same typed plans with a boundary-residual trie: choose the outer subject/verb/object words jointly, then permit only lexical repairs that preserve complete clause valency on both sides; log every repair and rerun this audit.",
        "scope": "A zero-pair result is diagnostic only; programmatic checks never certify readability.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"clauses": result["enumerated_clause_count"], "pairs": len(result["pairs"]),
                      "exact": result["exact_pair_count"]}, indent=2))


if __name__ == "__main__":
    main()
