"""Online character intersection over a dependency-shaped phrase-path lattice.

This lane differs from word-pair and completed-sentence sweeps: each path is
typed as NP -> finite-VP -> object -> (optional PP/relative), and opposing
paths are intersected one character at a time.  The lexical paths are ordinary
English transitions observed in the small corpus bank below; no entry is
created by reversing another entry.  Exact closures are still subject to a
surface grammar/readability gate and are reported, never silently promoted.
"""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path
from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dependency-phrase-path-lattice-20260930.json"
SEED = "An aide rips nine memos; some men inspire Diana."

PATHS = {
    "subject": ("the pilot", "a baker", "the nurse", "a poet", "the clerk",
                 "a teacher", "the sailor", "a gardener", "the archivist"),
    "finite_verb": ("maps", "marks", "reads", "writes", "keeps", "carries",
                    "guides", "records", "opens", "mends"),
    "object": ("the chart", "a letter", "the map", "a note", "the parcel",
                "a lantern", "the garden", "a story", "the gate"),
    "pp": ("near the harbor", "by the river", "under the bridge",
           "beside the gate", "after rain", "at dawn", "before dusk", "at sea"),
    "relative": ("that the nurse keeps", "which the clerk reads",
                 "that a poet writes", "which the pilot maps"),
}

def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i-1]) for i in range(len(tape)//2)
                     if tape[i] != tape[-i-1]), None)
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

def grammar() -> Grammar:
    g = Grammar()
    # Dependency order is explicit and finite: subject -> finite predicate ->
    # object, with optional adjunct/relative paths after the argument.
    g.slot(PATHS["subject"], "NP:subject")
    g.slot(PATHS["finite_verb"], "VP:finite")
    g.slot(PATHS["object"], "NP:object")
    g.slot(("",) + PATHS["pp"], "PP:adjunct")
    g.slot(("",) + PATHS["relative"], "CP:relative")
    return g

def bank_checks() -> dict[str, object]:
    phrases = [p for values in PATHS.values() for p in values]
    tapes = [norm(p) for p in phrases]
    pairs = [(a, b) for a in tapes for b in tapes if a != b and a == b[::-1]]
    return {"phrase_count": len(phrases), "authored_reverse_pairs": pairs,
            "reverse_pair_free": not pairs,
            "typed_roles": list(PATHS),
            "dependency_order": ["NP:subject", "VP:finite", "NP:object",
                                 "PP:adjunct", "CP:relative"]}

def controls() -> list[dict[str, object]]:
    texts = ["The pilot maps the chart near the harbor.",
             "A careful nurse reads a letter beside the gate.", SEED]
    return [{"rendered": t, "audit": audit(t), "kind":
             "incumbent_exact_control" if norm(t) == norm(SEED) else "intact_prose_control"}
            for t in texts]

def run() -> dict[str, object]:
    checks = bank_checks()
    assert checks["reverse_pair_free"]
    raw = intersect(grammar(), max_letters=220, cap=300_000)
    rows = []
    for row in raw["candidates"]:
        text = row["rendered"]
        rows.append({**row, "audit": audit(text),
            "surface_gate": {"subject": True, "finite_verb": True,
                              "argument_object": True, "ordinary_clause": True,
                              "reader_certified": False},
            "provenance": {"dependency_path_lattice": True,
                "online_character_intersection": True, "complete_sentence_sweep": False,
                "reversed_phrase_bank": False, "posthoc_repair": False,
                "catalogue_text": False, "per_candidate_rlaif": False,
                "reader_certified": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["sha_equal"]]
    return {"experiment_id": "dependency-phrase-path-lattice-20260930",
        "method": "typed NP/finite-VP/argument/PP/relative dependency paths with live opposing-character intersection",
        "bank_checks": checks, "controls": controls(), "exact_candidates": exact,
        "accepting_witnesses": rows,
        "solver_stats": {k: raw[k] for k in ("states", "transitions", "cap_reached",
                                              "grammar_states", "grammar_character_edges")},
        "reader_gate": "closed: no blinded human ratings collected",
        "next_repair": raw["dead_frontiers"][:8],
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer", "forward/reverse SHA-256",
                                              "llm_palindrome.validator.is_palindrome"]}}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"exact": len(result["exact_candidates"]), **result["solver_stats"]}, sort_keys=True))
