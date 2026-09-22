#!/usr/bin/env python3
"""Deterministic discourse-connectivity diagnostic for paragraph ABBA packets.

This is a filter only.  A pass means that the cheap adjacency test found a
possible bridge; it does not certify readability, coherence, or grammar.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

INPUT = Path("runs/paragraph-abba-boundary-carrier-20261002.json")
OUTPUT = Path("runs/paragraph-discourse-connectivity-gate-20261002.json")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PRONOUNS = {"he", "she", "they", "them", "him", "her", "it", "this", "that", "these", "those", "there"}
STOP = {"a", "an", "and", "at", "before", "by", "for", "from", "in", "of", "on", "or", "the", "then", "to", "was", "were", "with"}
EVENT_WORDS = {"saw", "opened", "marked", "closed", "walked", "carried", "watched", "moved", "found", "sent"}


def tokens(sentence: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(sentence)]


def evidence(left: str, right: str) -> dict:
    lt, rt = tokens(left), tokens(right)
    shared = sorted((set(lt) & set(rt)) - STOP)
    typed = []
    if shared:
        if any(x in EVENT_WORDS for x in shared):
            typed.append("shared_event")
        else:
            typed.append("shared_participant_or_object")
    # Explicit, surface anaphora is intentionally narrow and deterministic.
    anaphora = sorted(set(rt) & PRONOUNS)
    if anaphora:
        typed.append("typed_anaphora")
    return {"shared_tokens": shared, "anaphoric_tokens": anaphora,
            "relation_types": typed, "connected": bool(typed)}


def inspect(item: dict) -> dict:
    units = item.get("units") or re.split(r"(?<=[.!?])\s+", item["rendered"].strip())
    edges = [{"left_index": i, "right_index": i + 1,
              "left": units[i], "right": units[i + 1],
              **evidence(units[i], units[i + 1])}
             for i in range(len(units) - 1)
             for _ in [0]]
    exact = item.get("kind") == "exact_abba_candidate"
    return {"id": item["id"], "kind": item.get("kind"),
            "decision": ("retain" if all(e["connected"] for e in edges) else "reject_disconnected") if exact else "control_preserved",
            "filter_applied": exact, "edge_evidence": edges,
            "note": "Diagnostic filter only; no readability or coherence certification."}


def main() -> None:
    packet = json.loads(INPUT.read_text())
    # Prefer the de-duplicated candidate/control inventory when present.
    items = packet["items"]
    results = [inspect(item) for item in items]
    candidates = [r for r in results if r["filter_applied"]]
    controls = [r for r in results if not r["filter_applied"]]
    # Assertions are deliberately executable provenance checks.
    assert len(candidates) == packet["stats"]["exact_abba_candidates"]
    assert len(controls) == packet["stats"]["controls"]
    assert all(r["decision"] == "reject_disconnected" for r in candidates)
    assert all(r["decision"] == "control_preserved" for r in controls)
    synthetic = inspect({"id": "synthetic-connected", "kind": "exact_abba_candidate",
                         "units": ["Nora opened the archive.", "She marked the map."]})
    assert synthetic["decision"] == "retain" and synthetic["edge_evidence"][0]["relation_types"] == ["typed_anaphora"]
    out = {"experiment_id": "paragraph-discourse-connectivity-gate-20261002",
           "method": "deterministic adjacent-unit lexical and typed-anaphora gate",
           "input": str(INPUT), "source_experiment_id": packet["experiment_id"],
           "scope": "diagnose/filter exact ABBA candidates; preserve controls; never certify readability",
           "rule": "Every adjacent candidate pair must share a non-stopword participant/object/event token or have an explicit pronoun/demonstrative anaphor in the right unit.",
           "results": results,
           "summary": {"candidate_count": len(candidates),
                       "candidate_rejected": sum(r["decision"] == "reject_disconnected" for r in candidates),
                       "controls_preserved": len(controls)},
           "provenance": {"lm": False, "rlaif": False, "input_sha256": __import__("hashlib").sha256(INPUT.read_bytes()).hexdigest(),
                          "assertions": ["candidate/control counts", "all packet candidates reject", "all controls preserved", "synthetic anaphora retain"]},
           "next_repair": "Author explicit participant/object/event bridges or typed anaphora between every adjacent candidate sentence, then rerun this gate; a retained result still requires human readability review."}
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
