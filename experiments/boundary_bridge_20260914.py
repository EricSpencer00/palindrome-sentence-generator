"""Bounded, reader-first constituent-pair boundary bridge experiment.

The bridge search changes the first conflicting constituent on *both* sides,
then hands the witnessed residual to the ordinary whole-discourse solver.  It
never emits a clause fragment or claims that exactness implies readability.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.discourse_subset_word_equation_20260914 import independent_audit, letters

@dataclass(frozen=True)
class Constituent:
    slot: str
    kind: str
    options: tuple[str, ...]

@dataclass(frozen=True)
class Clause:
    ident: str
    constituents: tuple[Constituent, ...]
    def render(self, choice: tuple[int, ...]) -> str:
        return " ".join(c.options[n] for c, n in zip(self.constituents, choice)) + "."

SCENE = (
    Clause("left", (Constituent("subject", "agent", ("Mara", "The nurse", "A")),
                     Constituent("verb", "transitive", ("checks", "watches")),
                     Constituent("object", "patient", ("the quiet child", "the small boat")))),
    Clause("right", (Constituent("subject", "agent", ("The child", "A sailor", "A")),
                      Constituent("verb", "transitive", ("rests", "guides")),
                      Constituent("object", "patient", ("near Mara", "the nurse")))),
)
CONTINUATIONS = ("The lantern warms the room.", "A quiet student reads.",
                 "Fresh paper covers the desk.", "The teacher closes the door.")

def bridge(left: Clause, right: Clause, limit: int = 256) -> list[dict]:
    """Try jointly rewritten complete clauses; retain only nonempty residuals."""
    out = []
    import itertools
    for lc in itertools.product(*(range(len(c.options)) for c in left.constituents)):
        for rc in itertools.product(*(range(len(c.options)) for c in right.constituents)):
            lt, rt = letters(left.render(lc)), letters(right.render(rc))
            n = min(len(lt), len(rt))
            common = 0
            while common < n and lt[common] == rt[::-1][common]: common += 1
            if common == 0 or common == n and len(lt) == len(rt): continue
            residual = (lt[common:] if len(lt) >= len(rt) else rt[::-1][common:])
            if not residual: continue
            out.append({"left": left.render(lc), "right": right.render(rc),
                        "choices": {"left": lc, "right": rc}, "cancelled_letters": common,
                        "residual": residual, "residual_owner": "left" if len(lt) >= len(rt) else "right"})
            if len(out) >= limit: return out
    return out

def resume_from_witness(witness: dict, clauses: tuple[str, ...], *, min_letters: int = 39,
                        max_letters: int = 200,
                        max_continuations: int = 3) -> list[tuple[int, ...]]:
    """Close a witnessed outer pair with complete clauses between its edges."""
    results = []
    for count in range(min(len(clauses), max_continuations) + 1):
        for ids in itertools.permutations(range(len(clauses)), count):
            middle = tuple(clauses[i] for i in ids)
            text = " ".join((witness["left"],) + middle + (witness["right"],))
            tape = letters(text)
            if min_letters <= len(tape) <= max_letters and tape == tape[::-1]:
                results.append(ids)
    return results

def run() -> dict:
    witnesses = bridge(*SCENE)
    records = []
    trials = 0
    # Resume the discourse solver from each witnessed, fully rendered pair.
    for witness_index, witness in enumerate(witnesses):
        pair_found = resume_from_witness(witness, CONTINUATIONS, min_letters=39)
        trials += sum(
            len(tuple(itertools.permutations(range(len(CONTINUATIONS)), count)))
            for count in range(4)
        )
        for ids in pair_found:
            text = " ".join((witness["left"],) + tuple(CONTINUATIONS[i] for i in ids) + (witness["right"],))
            audit = independent_audit(text)
            records.append({"witness_index": witness_index, "continuation_indices": ids,
                            "text": text,
                            "normalized": audit["normalized"], "audit": audit,
                            "reader_worthy": False, "rejection": "no reader evidence"})
    return {"status": "complete_boundary_bridge_experiment", "config": {"pair_limit": 256, "min_letters": 39},
            "witnesses": witnesses,
            "search": {"witnesses": len(witnesses), "continuation_orders_tested": trials,
                       "max_continuations": 3},
            "exact_closures": len(records), "records": records,
            "provenance": "Fresh authored same-scene clauses with typed valency; no catalogue material.",
            "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "next_operator_if_empty": "Expand the typed transitive-verb/object alternatives at the first residual seam, then rerun whole-discourse subset search."}

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.out.exists(): p.error("refusing to overwrite existing output")
    result = run(); a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(a.out), "witnesses": len(result["witnesses"]), "exact_closures": result["exact_closures"]}))
if __name__ == "__main__": main()
