"""Indexed reverse-segmentation search over typed ordinary clauses.

The search constructs grammatical clauses first.  It then indexes their
letter tapes and looks up the exact reverse tape as a *different* grammatical
clause; no language model or per-candidate reward is involved.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "reverse-segmentation-clause-index-20260919"


def _load_bank():
    import importlib.util
    spec = importlib.util.spec_from_file_location("half_bank", ROOT / "experiments" / "half_tape_grammar_csp_20260919.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clauses(bank):
    rows = []
    for subj, verb, obj in itertools.product(bank.SUBJECTS, bank.VERBS, bank.OBJECTS):
        if subj.number != verb.number or obj.object_type != verb.object_type:
            continue
        if subj.content & verb.content or subj.content & obj.content or verb.content & obj.content:
            continue
        suffixes = ((),) + tuple(x.words for x in bank.ADJUNCTS)
        for suffix in suffixes:
            words = subj.words + verb.words + obj.words + suffix
            text = " ".join(words)
            rows.append({"text": text, "tape": normalize_letters(text), "words": words})
    return rows


def _audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    digest = hashlib.sha256(tape.encode()).hexdigest()
    reverse_digest = hashlib.sha256(tape[::-1].encode()).hexdigest()
    two_pointer = all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2))
    checks = mechanical_admission_checks(text, min_letters=30)
    mechanical = all(bool(value) for value in checks.values())
    return {"length": len(tape), "two_pointer": two_pointer,
            "sha256": digest, "reverse_sha256": reverse_digest,
            "sha_match": digest == reverse_digest, "mechanical": mechanical,
            "checks": checks}


def run() -> dict[str, object]:
    bank = _load_bank()
    clauses = _clauses(bank)
    index = {row["tape"]: row for row in clauses}
    matches = []
    for left in clauses:
        right = index.get(left["tape"][::-1])
        if right is None or left["text"] == right["text"]:
            continue
        rendered = left["text"].capitalize() + "; " + right["text"] + "."
        audit = _audit(rendered)
        matches.append({"rendered": rendered, "left": left["text"], "right": right["text"], "provenance": EXPERIMENT_ID, "audit": audit})
    matches.sort(key=lambda row: row["audit"]["length"], reverse=True)
    return {"experiment": EXPERIMENT_ID, "method": "typed clause generation followed by indexed reverse-tape resegmentation", "clause_count": len(clauses), "reverse_matches": len(matches), "candidates": matches[:50], "next_repair": "index partial reverse prefixes and add held-out Shakespearean transitive frames; retain exact closure and independent audit gates"}


if __name__ == "__main__":
    out = run()
    out_path = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("experiment", "clause_count", "reverse_matches", "next_repair")}, indent=2))
    for row in out["candidates"][:5]:
        print(row["rendered"], row["audit"])
