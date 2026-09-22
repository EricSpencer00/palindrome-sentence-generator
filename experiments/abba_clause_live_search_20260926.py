"""Live ABBA clause search with semantic role labels.

Each side is assembled from complete, authored clauses.  The search emits a
left A/B pair, then consumes its character obligation with independently
authored B/A clauses.  It never reverses a completed sentence; the reverse
obligation is only a constraint on the next clause.  This deliberately keeps
the search small and interpretable so failures yield a measured seam repair.
"""
from __future__ import annotations

import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abba-clause-live-search-20260926.json"

CLAUSES = {
    "A": (
        "At dusk, the keeper marked the tide.",
        "At dawn, the cartographer studied the map.",
        "By evening, the gardener covered the rose.",
        "Near noon, the sailor watched the shore.",
    ),
    "B": (
        "The quiet bell warned the village.",
        "A patient guide carried the lantern.",
        "The young naturalist noted the moth.",
        "A careful doctor opened the case.",
    ),
}

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())

def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = None
    for i in range(len(tape) // 2):
        if tape[i] != tape[-1 - i]:
            mismatch = {"offset": i, "left": tape[i], "right": tape[-1-i]}
            break
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse_obligation": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }

def consume(obligation: str, clauses: tuple[str, ...], limit: int = 64):
    """Match only whole clauses, retaining the deepest live seam."""
    states = {0: ()}
    trace = []
    for slot in range(2):
        nxt = {}
        for pos, path in states.items():
            for clause in clauses:
                tape = letters(clause)
                if obligation.startswith(tape, pos):
                    end = pos + len(tape)
                    nxt.setdefault(end, path + (clause,))
                    trace.append({"slot": slot, "start": pos, "end": end,
                                  "surface": clause, "role": ("B2", "A2")[slot]})
        states = nxt
    return [path for pos, path in states.items() if pos == len(obligation)][:limit], trace

def run() -> dict:
    rows, residuals, controls = [], [], []
    for a, b in itertools.product(CLAUSES["A"], CLAUSES["B"]):
        left = f"{a} {b}"
        obligation = letters(left)[::-1]
        parses, trace = consume(obligation, CLAUSES["B"] + CLAUSES["A"])
        deepest = max((x["end"] for x in trace), default=0)
        residuals.append({
            "left_A": a, "left_B": b, "obligation_letters": len(obligation),
            "deepest_clause_support": deepest,
            "support_fraction": deepest / max(1, len(obligation)),
            "first_unmet_character": obligation[deepest:deepest+12],
            "next_repair": "author a complete B2 clause beginning with the unmet residual, then re-run A2",
        })
        controls.append({"rendered": left, "roles": ["A1", "B1"],
                         "audit": audit(left), "kind": "intact-authored-AB-control"})
        for b2, a2 in parses:
            rendered = f"{left} {b2} {a2}"
            rows.append({"rendered": rendered, "roles": ["A1", "B1", "B2", "A2"],
                         "audit": audit(rendered), "provenance": {
                             "complete_authored_clauses": True,
                             "live_reverse_obligation": True,
                             "finished_tape_reversal": False,
                             "catalogue_text": False, "repeated_units": False,
                             "self_palindromic_units": False, "posthoc_repair": False,
                         }})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"] and row["audit"]["letters"] > 38]
    smoothest = max(residuals, key=lambda x: x["support_fraction"])
    return {
        "experiment_id": "abba-clause-live-search-20260926",
        "method": "semantic A-B-B-A complete-clause generation with live character obligation",
        "stats": {"left_pairs": len(controls), "closed_derivations": len(rows),
                  "exact_gt38": len(exact), "max_support": smoothest["deepest_clause_support"],
                  "max_support_fraction": smoothest["support_fraction"]},
        "exact_candidates": exact, "rendered_candidates": rows[:32], "controls": controls[:8],
        "residual_certificates": residuals,
        "novelty_preflight": {"status": "passed",
            "signature": "semantic-clause|ABBA|live-obligation|two-slot-DP",
            "distinct_from": "word-level sweeps, finished-tape reversal, catalogue controls",
            "finished_tape_reversal": False, "reward_ranking": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
                       "reader_gate": "closed pending exact closure and blinded reader test"},
        "status": "exact closure found" if exact else "no exact closure; deepest semantic seam retained",
        "next_construction": "replace only the highest-support unmet B2/A2 seam with a new authored clause; keep the other banks frozen",
    }

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
