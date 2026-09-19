"""Deterministic two-frontier, typed half-tape pilot; no language-model calls.

Variables are mirrored character orbits, with a single grammar path whose
unassigned contiguous interior is expanded from its most constrained edge.
Verb/subject/object features propagate regardless of assignment order.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS, mechanical_admission_checks, normalize_letters,
)

ID = "astra-bidirectional-half-tape-20260920"


@dataclass(frozen=True)
class Edge:
    text: str
    number: str = ""
    kind: str = ""
    proper: bool = False

    @property
    def tape(self):
        return normalize_letters(self.text)

    @property
    def content(self):
        return frozenset(w.lower() for w in self.text.split()
                         if w.lower() not in REPEATABLE_FUNCTION_WORDS)


def bank():
    people = "aide bard poet scribe sailor nurse baker writer clerk man woman child singer reader guard teacher friend brother sister lord lady".split()
    plurals = "aides bards poets scribes sailors nurses bakers writers clerks men women children singers readers guards teachers friends brothers sisters lords ladies".split()
    subjects = []
    for noun in people:
        article = "an" if noun[0] in "aeiou" else "a"
        subjects.extend(Edge(f"{d} {noun}", "sg", "person") for d in (article, "the", "one"))
    for noun in plurals:
        subjects.extend(Edge(f"{d} {noun}", "pl", "person") for d in ("some", "the", "nine"))
    names = [Edge(n, "sg", "person", True) for n in "Diana Nora Leon Ada Eve Iris".split()]
    subjects += names
    documents = []
    for noun in "memo note letter map chart page poem plan book song tale story sonnet".split():
        documents.extend(Edge(f"{d} {noun}", "sg", "document") for d in ("a", "the", "one"))
    for noun in "memos notes letters maps charts pages poems plans books songs tales stories sonnets".split():
        documents.extend(Edge(f"{d} {noun}", "pl", "document") for d in ("some", "nine", "the"))
    verbs = []
    for kind, pairs in {
        "document": "rip:rips read:reads mark:marks write:writes keep:keeps find:finds save:saves tear:tears carry:carries lose:loses mend:mends sign:signs send:sends",
        "person": "inspire:inspires guide:guides praise:praises help:helps lead:leads meet:meets love:loves trust:trusts warn:warns teach:teaches greet:greets see:sees hear:hears",
    }.items():
        for pair in pairs.split():
            pl, sg = pair.split(":")
            verbs.extend((Edge(pl, "pl", kind), Edge(sg, "sg", kind)))
    return {
        "S": tuple(subjects), "V": tuple(verbs),
        "O": tuple(documents + subjects),
        "A": tuple(Edge(w) for w in "now often once still never again soon".split()),
        "C": (Edge("as"), Edge("while"), Edge("and"), Edge("but")),
        "B": (Edge(";"),),
    }


DOMAINS = bank()
# Each clause contains a full, typed S V O path. Adverbs are grammatically
# placed, and punctuation is fixed before search; no post-hoc fragment repair.
FRAMES = {
    "joined": ("S0", "V0", "O0", "C", "S1", "V1", "O1"),
    "left_adverb": ("S0", "A", "V0", "O0", "C", "S1", "V1", "O1"),
    "right_adverb": ("S0", "V0", "O0", "C", "S1", "A", "V1", "O1"),
    "two_clauses_left_adverb": ("S0", "A", "V0", "O0", "B", "S1", "V1", "O1"),
    "two_clauses_right_adverb": ("S0", "V0", "O0", "B", "S1", "A", "V1", "O1"),
}


def audit(text, minimum=39):
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    mismatch = None
    while i < j:
        if tape[i] != tape[j]:
            mismatch = (i, j, tape[i], tape[j])
            break
        i += 1
        j -= 1
    gates = mechanical_admission_checks(text, min_letters=minimum, max_letters=52)
    return {"text": text, "letters": len(tape), "words": len(text.replace(";", "").split()),
            "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "mechanical_checks": gates, "mechanically_admitted": all(gates.values()),
            "reader_status": "unreviewed"}


def solve(frame, target, budget=100_000):
    slots = tuple(DOMAINS[s[0]] for s in frame)
    metadata = {e: (e.tape, e.content, len(e.text.split()) if e.text != ";" else 0)
                for domain in slots for e in domain}
    minlen = [min(len(metadata[e][0]) for e in d) for d in slots]
    maxlen = [max(len(metadata[e][0]) for e in d) for d in slots]
    stats = {"edge_attempts": 0, "nodes": 0, "character_conflicts": 0,
             "feature_conflicts": 0, "length_prunes": 0, "budget_exhausted": False}
    solutions = []

    def compatible(index, edge, selected):
        slot = frame[index]
        if slot[0] not in "SVO":
            return True
        group = slot[-1]
        values = {frame[k][0]: v for k, v in selected.items() if frame[k][-1] == group}
        values[slot[0]] = edge
        if "S" in values and "V" in values and values["S"].number != values["V"].number:
            return False
        if "V" in values and "O" in values and values["V"].kind != values["O"].kind:
            return False
        return True

    def candidates(index, start, end, assigned, selected, used, proper, wc, from_right):
        result = []
        for edge in slots[index]:
            if stats["edge_attempts"] >= budget:
                stats["budget_exhausted"] = True
                break
            stats["edge_attempts"] += 1
            tape, content, count = metadata[edge]
            if used & content or proper + edge.proper > 1 or wc + count > 15:
                continue
            if not compatible(index, edge, selected):
                stats["feature_conflicts"] += 1
                continue
            pos = end - len(tape) if from_right else start
            if pos < start or pos + len(tape) > end:
                continue
            updates = {}
            for j, char in enumerate(tape):
                alias = min(pos + j, target - 1 - pos - j)
                prior = updates.get(alias, assigned.get(alias))
                if prior is not None and prior != char:
                    stats["character_conflicts"] += 1
                    break
                updates[alias] = char
            else:
                result.append((edge, updates, len(tape), content, count))
        return result

    def visit(lo, hi, start, end, assigned, selected, used, proper, wc):
        if stats["budget_exhausted"] or len(solutions) >= 20:
            return
        stats["nodes"] += 1
        if lo > hi:
            if start == end and 8 <= wc <= 15:
                text = " ".join(selected[k].text for k in range(len(frame))).replace(" ;", ";")
                text = text[0].upper() + text[1:] + "."
                row = audit(text, minimum=min(39, target))
                row["selected_path"] = [selected[k].text for k in range(len(frame))]
                solutions.append(row)
            return
        if not sum(minlen[lo:hi+1]) <= end-start <= sum(maxlen[lo:hi+1]):
            stats["length_prunes"] += 1
            return
        left = candidates(lo, start, end, assigned, selected, used, proper, wc, False)
        if not left:
            return
        right = candidates(hi, start, end, assigned, selected, used, proper, wc, True) if lo != hi else left
        if not right:
            return
        take_right = len(right) < len(left)
        index = hi if take_right else lo
        for edge, updates, length, content, count in right if take_right else left:
            visit(lo + (not take_right), hi - take_right,
                  start + (0 if take_right else length), end - (length if take_right else 0),
                  assigned | updates, selected | {index: edge}, used | content,
                  proper + edge.proper, wc + count)

    visit(0, len(frame)-1, 0, target, {}, {}, frozenset(), 0, 0)
    return solutions, stats


def run():
    rows, trace = [], []
    baseline, baseline_stats = solve(("S0", "V0", "O0", "B", "S1", "V1", "O1"), 38, 1_000_000)
    for length in range(39, 53):
        for name, frame in FRAMES.items():
            found, stats = solve(frame, length)
            for row in found:
                row["frame"] = name
            rows.extend(found)
            trace.append({"target": length, "frame": name, "exact": len(found), **stats})
        print(json.dumps({"finished_length": length, "exact_so_far": len(rows)}), flush=True)
    unique = {r["sha256_forward"]: r for r in rows}
    result = {"experiment_id": ID, "representation": "two-frontier typed half-tape CSP",
              "runtime": "deterministic Python, no model or remote API",
              "baseline": baseline, "baseline_stats": baseline_stats,
              "target_lengths": list(range(39, 53)), "word_range": [8, 15],
              "edge_attempt_budget_per_cell": 100_000, "trace": trace,
              "candidates": list(unique.values()),
              "stats": {"edge_attempts": sum(t["edge_attempts"] for t in trace),
                        "nodes": sum(t["nodes"] for t in trace),
                        "budget_exhausted_cells": sum(t["budget_exhausted"] for t in trace),
                        "exact_tapes": len(unique),
                        "mechanically_admitted": sum(r["mechanically_admitted"] for r in unique.values())},
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "vocabulary_sha256": hashlib.sha256(repr(DOMAINS).encode()).hexdigest(),
              "vocabulary_words": sorted({w.lower() for d in DOMAINS.values() for e in d for w in e.text.split()}),
              "novelty_scope": "local catalogue and construction gates only; no claim of global originality",
              "next_repair": "Use conflict counts and surviving frontier domains to choose a new clause relation; do not infer impossibility from exhausted cells."}
    out = ROOT / "runs" / f"{ID}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    return result


if __name__ == "__main__":
    run()
