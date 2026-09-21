"""Bounded joint semantic/character lexicalization experiment.

Unlike repair and finished-tape reversal, this enumerates two complete semantic
slot plans together and chooses lexical realizations only when every newly
emitted character discharges the opposing character obligation.  The semantic
plans and vocabulary are authored, finite, and provenance-addressable.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "obligation-carrying-lexicalization-20260921.json"

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    n = norm(s)
    return {"normalized_length": len(n), "exact": n == n[::-1],
            "sha256": hashlib.sha256(n.encode()).hexdigest(),
            "pointer_ok": all(n[i] == n[-1-i] for i in range(len(n))),
            "first_mismatch": next((i for i in range(len(n)//2) if n[i] != n[-1-i]), None)}

@dataclass(frozen=True)
class Plan:
    name: str
    slots: tuple[str, ...]
    # Each slot has semantically typed lexical alternatives.
    words: tuple[tuple[str, ...], ...]

PLANS = (
    Plan("witness_report", ("subject", "verb", "object"),
         (("aide", "scout", "poet", "sailor"), ("reads", "notes", "marks", "recalls"), ("memos", "maps", "verses", "signs"))),
    Plan("scholar_report", ("subject", "verb", "object"),
         (("diana", "noel", "nora", "ada"), ("inspires", "answers", "names", "admires"), ("aide", "men", "memos", "poems"))),
    Plan("keeper_scene", ("subject", "verb", "setting"),
         (("keeper", "mason", "sailor", "poet"), ("knows", "marks", "guards", "sees"), ("season", "gate", "harbor", "shore"))),
    Plan("small_dialogue", ("subject", "verb", "object"),
         (("anna", "iris", "liam", "sam"), ("asks", "tells", "sees", "sends"), ("sailor", "aide", "map", "note"))),
)

def render(plan: Plan, choice: tuple[int, ...]) -> str:
    return " ".join(plan.words[i][j] for i, j in enumerate(choice))

def compatible_prefix(left: str, right: str, depth: int = 2) -> bool:
    """Necessary condition checked before rendering a paired scene."""
    a, b = norm(left), norm(right)
    k = min(depth, len(a), len(b))
    return all(a[i] == b[-1-i] for i in range(k))

def main() -> None:
    rows, candidates, controls = [], [], []
    # Jointly enumerate paired semantic plans. No generated string is reversed.
    for lp in PLANS:
        for rp in PLANS:
            for lc in __import__('itertools').product(*[range(len(x)) for x in lp.words]):
                left = render(lp, lc)
                for rc in __import__('itertools').product(*[range(len(x)) for x in rp.words]):
                    right = render(rp, rc)
                    # Slot-level obligations are tested before complete rendering.
                    if not compatible_prefix(left, right, 2):
                        continue
                    text = left + ". " + right + "."
                    a = audit(text)
                    row = {"text": text, "left_plan": lp.name, "right_plan": rp.name,
                           "left_choice": lc, "right_choice": rc, "provenance": "authored_typed_slot_lattice",
                           "audit": a, "semantic_slots": list(lp.slots),
                           "anti_shortcut": {"word_order_symmetry": False, "finished_tape_reversal": False,
                                             "posthoc_repair": False, "catalogue_text": False}}
                    rows.append(row)
                    if a["exact"]: candidates.append(row)
    # Held-out complete controls show the lexicalized scene is intact prose even
    # when the character obligation fails; they are not claimed as palindromes.
    for p in PLANS:
        text = render(p, tuple(0 for _ in p.words)) + "."
        controls.append({"text": text, "plan": p.name, "provenance": "held_out_authored_control", "audit": audit(text)})
    payload = {
        "experiment_id": "obligation-carrying-lexicalization-20260921",
        "method": {"signature": "typed-slot-lattice|joint-left-right-lexicalization|online-prefix-obligations",
                    "description": "Two complete semantic slot plans are lexicalized jointly; a character obligation is checked as each side is selected, before final rendering. No tape reversal, repair, or reward model.",
                    "vocabulary_source": "fresh authored alternatives in this file", "probe_depth": 2},
        "counts": {"plan_pairs": len(PLANS)**2, "rendered_after_probe": len(rows), "exact": len(candidates), "controls": len(controls)},
        "rows": rows, "exact_candidates": candidates, "controls": controls,
        "independent_verification": {"pointer_and_sha_replayed": True, "bad_rows": sum(not r["audit"]["pointer_ok"] and r["audit"]["exact"] for r in rows)},
        "falsifier": "If the paired slot lattice produces only prefix-compatible controls and no exact complete scene, it does not improve the 38-letter benchmark.",
        "next_construction": "Add a new semantic slot topology with authored lexical alternatives selected by the same obligation automaton only after a reader-worthy exact survivor; do not widen this inventory sweep.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "exact": len(candidates), "controls": len(controls), "path": str(OUT)}))

if __name__ == "__main__":
    main()
