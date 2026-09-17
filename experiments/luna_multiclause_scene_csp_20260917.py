"""Multi-clause scene CSP: semantic valency plus joint boundary equations.

This lane composes two ordinary clauses per side (coordination/causality), then
solves only their exposed character obligations jointly.  It never reverses a
finished sentence or mirrors word order; an exact closure, if found, is still
only eligible for later human study.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-multiclause-scene-csp-20260917.json"
ID = "luna-multiclause-scene-csp-20260917"
SIG = "multiclause-scene-csp|coordinated-causal-valency|joint-boundary-equations|direct-reverse-opposing-index-audit"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

@dataclass(frozen=True)
class Clause:
    subject: str; verb: str; object: str; adjunct: str; relation: str = ""
    def render(self) -> str:
        return f"The {self.subject} {self.verb} the {self.object} {self.adjunct}"
    def roles(self) -> dict[str, str]:
        return {"agent": self.subject, "event": self.verb, "theme": self.object, "adjunct": self.adjunct}

SUBJECTS = ("careful baker", "quiet gardener", "patient teacher", "young sailor")
VERBS = ("carries", "checks", "opens", "records")
OBJECTS = ("bright lantern", "sealed letter", "small basket", "weathered map")
ADJUNCTS = ("before dusk", "near the harbor", "beside the garden", "after the rain")

def clauses() -> tuple[Clause, ...]:
    # Semantic valency is explicit: all are transitive SVO + licensed adjunct.
    return tuple(Clause(s, v, o, a) for s, v, o, a in itertools.product(SUBJECTS, VERBS, OBJECTS, ADJUNCTS))

def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); n = len(tape)
    opposing = [{"left_index": i, "right_index": n-1-i, "left": tape[i], "right": tape[n-1-i], "match": tape[i] == tape[n-1-i]} for i in range(n//2)]
    return {"letters": n, "normalized_tape": tape, "exact": bool(tape) and tape == tape[::-1],
            "direct_reverse_exact": tape == tape[::-1], "opposing_index_exact": all(x["match"] for x in opposing),
            "opposing_index_mismatches": [x for x in opposing if not x["match"]][:12],
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "mechanical_checks": mechanical_admission_checks(text, min_letters=39, max_letters=240)}

def novelty_preflight() -> dict[str, object]:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    artifact = str(Path(__file__).relative_to(ROOT))
    result = {"status": "passed", "registry_entries_read": len(entries),
              "signature_overlaps": [e.get("signature") for e in entries if e.get("signature") == SIG],
              "artifact_collisions": [e.get("artifact") for e in entries if e.get("artifact") == artifact],
              "rejected_shortcuts": ["finished-tape reversal", "word-order mirror", "catalogue text", "fragment", "gibberish"]}
    if result["signature_overlaps"] or result["artifact_collisions"]: raise RuntimeError(result)
    return result

def candidate(left: tuple[Clause, Clause], right: tuple[Clause, Clause]) -> dict[str, object]:
    # Each side remains ordinary order; only exposed edge equations are scored.
    text = ". ".join((left[0].render(), left[1].render(), right[0].render(), right[1].render())) + "."
    words = tokenize(text)
    equations = [{"left_clause": i, "right_clause": 1-i, "left_edge": normalize_letters(c.render())[-1],
                  "right_edge": normalize_letters(right[1-i].render())[0], "satisfied": normalize_letters(c.render())[-1] == normalize_letters(right[1-i].render())[0]}
                 for i, c in enumerate(left)]
    a = audit(text)
    return {"rendered": text, "clauses": [[c.roles() for c in left], [c.roles() for c in right]],
            "joint_boundary_equations": equations, "audit": a, "length": a["letters"],
            "provenance": {"lexical_source": "fresh hand-authored valency inventory", "catalogue_text_imported": False,
                           "word_order_mirror": False, "finished_tape_reversal": False, "semantic_valency": "transitive SVO + temporal/locative adjunct"},
            "reader_eligible": bool(a["exact"] and a["letters"] > 38),
            "reader_status": "unreviewed; human study required after all mechanical gates"}

def run() -> dict[str, object]:
    preflight = novelty_preflight(); bank = clauses()
    # Independent semantic selections; bounded CSP keeps the run reproducible.
    lefts = [(bank[i], bank[(i*7+3) % len(bank)]) for i in range(0, len(bank), max(1, len(bank)//10))][:10]
    rights = [(bank[-1-i], bank[-1-((i*11+5) % len(bank))]) for i in range(10)]
    rows = [candidate(l, r) for l, r in itertools.product(lefts, rights)]
    rows.sort(key=lambda x: (x["audit"]["exact"], sum(e["satisfied"] for e in x["joint_boundary_equations"]), x["length"]), reverse=True)
    exact = [r for r in rows if r["reader_eligible"]]
    return {"experiment_id": ID, "signature": SIG, "status": "completed_exact" if exact else "completed_no_exact_closure",
            "method": "two independently coherent coordinated/causal clauses per side; joint seam CSP with semantic valency",
            "novelty_preflight": preflight, "candidate_count": len(rows), "candidates": rows[:12],
            "stats": {"semantic_clause_bank": len(bank), "bounded_scene_pairs": len(rows), "exact": len(exact), "over_38_letters": sum(r["length"] > 38 for r in rows)},
            "controls": {"direct_reverse_audit": True, "opposing_index_audit": True, "controls_kept_separate": True},
            "failure_and_repair": {"next_operator": "hold the two-clause discourse frame fixed and add a held-out causal adjunct whose first/last characters satisfy the highest-scoring unsatisfied seam equation; replay the same CSP", "exact_closure_found": bool(exact)},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "fresh_authored_inventory": True, "human_review": "not run"}}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
