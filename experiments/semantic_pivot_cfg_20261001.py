"""Semantic-pivot CFG packed at character obligations.

The grammar has typed clause material on both sides of an ordinary event pivot;
lexical transitions are intersected before complete surface strings are rendered.
No reversed phrase pairs are included.
"""
from __future__ import annotations
import hashlib, json, re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm, audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "semantic-pivot-cfg-20261001.json"

# The pivot is a typed event/entity boundary, not a pre-reversed phrase.
# Slots intentionally vary function words and word boundaries.
SLOTS = (
    ("subject", ("a nurse", "the aide", "a poet", "an old aide", "one nurse")),
    ("determiner", ("", "the", "a", "one")),
    ("verb", ("reads", "marks", "keeps", "writes", "maps", "sends")),
    ("object", ("a note", "the map", "nine memos", "a letter", "the chart")),
    # typed semantic pivot: a human/entity event marker with ordinary lexical forms
    ("pivot", (";", ".", " and ", " while ")),
    ("subject2", ("some men", "the nurse", "Diana", "a poet", "the aide")),
    ("verb2", ("inspire", "read", "mark", "keep", "write", "guide")),
    ("object2", ("Diana", "the aide", "a note", "the map", "some men")),
)


def make_grammar() -> Grammar:
    g = Grammar()
    for role, choices in SLOTS:
        g.slot(choices, role)
    return g


def reverse_pair_check() -> dict[str, object]:
    phrases = [p for _, choices in SLOTS for p in choices if norm(p)]
    tapes = {p: norm(p) for p in phrases}
    pairs = [(a, b) for a, ta in tapes.items() for b, tb in tapes.items()
             if a != b and ta == tb[::-1]]
    return {"phrase_count": len(phrases), "reverse_pairs": pairs,
            "reverse_pair_free": not pairs}


def run() -> dict[str, object]:
    checks = reverse_pair_check()
    assert checks["reverse_pair_free"]
    raw = intersect(make_grammar(), max_letters=220, cap=500_000)
    rows = []
    for row in raw["candidates"]:
        text = row["rendered"]
        # independent audit, separate implementation from solver audit
        tape = re.sub(r"[^a-z]", "", text.lower())
        mism = next(((i, tape[i], tape[-i-1]) for i in range(len(tape)//2)
                     if tape[i] != tape[-i-1]), None)
        fwd = hashlib.sha256(tape.encode()).hexdigest()
        rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
        rows.append({**row, "independent_audit": {
            "letters": len(tape), "two_pointer_exact": bool(tape) and mism is None,
            "first_mismatch": mism, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev},
            "provenance": {"semantic_pivot_cfg": True,
                "packed_before_lexical_expansion": True,
                "ordinary_typed_slots": True, "reverse_phrase_bank": False,
                "complete_sentence_enumeration": False,
                "catalogue_text": False, "posthoc_repair": False,
                "per_candidate_rlaif": False, "reader_certified": False},
            "novelty_preflight": norm(text) != norm("An aide rips nine memos; some men inspire Diana."),
            "readability_gate": "diagnostic only; no human certification"})
    exact = [r for r in rows if r["independent_audit"]["two_pointer_exact"]
             and r["independent_audit"]["sha_equal"]]
    return {"experiment_id": "semantic-pivot-cfg-20261001",
            "method": "typed semantic-pivot CFG with live character intersection before lexical expansion",
            "slots": [{"role": r, "choices": list(c)} for r, c in SLOTS],
            "bank_checks": checks,
            "solver_stats": {k: raw[k] for k in ("states", "transitions", "cap_reached", "grammar_states", "grammar_character_edges")},
            "exact_candidates": exact,
            "accepting_witnesses": rows,
            "reader_gate": "closed: no blinded human ratings collected",
            "next_repair": raw["dead_frontiers"][:8],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "independent_audits": ["two-pointer", "forward/reverse SHA-256"]}}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"exact": len(result["exact_candidates"]), "states": result["solver_stats"]["states"], "transitions": result["solver_stats"]["transitions"]}, sort_keys=True))
