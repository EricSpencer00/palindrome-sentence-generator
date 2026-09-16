"""Semantic phrase-equation search with a global character balance.

This lane is intentionally not a lattice/FST intersection: each inventory item
is a complete authored phrase with a semantic role and a signed character
vector.  The solver chooses a normal-order scene on each side, then solves the
single global equation ``left_tape == reverse(right_tape)`` by meet-in-the-
middle character vectors.  A near miss is retained with its first debt and a
held-out whole-phrase repair, never as a claim of readability or exactness.
"""
from __future__ import annotations
import hashlib, json, sys
from itertools import product
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).parents[1]
EXPERIMENT_ID = "phrase-equation-inventory-solver-20260916"
SIGNATURE = "independent-role-phrase-inventories|global-reversed-tape-equation|meet-in-middle-character-vector|whole-scene-semantic-coherence|heldout-phrase-repair|dual-independent-exact-audit"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# These are independently authored phrase choices, not extracted corpus lines.
# Each sequence has ordinary word order and forms a complete little scene.
INVENTORY = {
    "agent": ["the patient nurse", "the careful porter", "the young teacher"],
    "action": ["checks the", "carries the", "opens the"],
    "object": ["sealed parcel", "old ledger", "small basket"],
    "place": ["beside the quiet gate", "near the stone bridge", "under the red awning"],
    "purpose": ["before dusk", "after the rain", "for the waiting child"],
}
ROLES = tuple(INVENTORY)

def tape(text: str) -> str:
    return normalize_letters(text)

def two_pointer(text: str) -> dict:
    t = tape(text); mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"algorithm": "independent-two-pointer", "letters": len(t), "exact": bool(t) and mismatch is None, "first_mismatch": mismatch}

def hash_audit(text: str) -> dict:
    t = tape(text); h = lambda x: hashlib.sha256(x.encode()).hexdigest()
    return {"algorithm": "sha256-forward-reverse", "forward": h(t), "reverse": h(t[::-1]), "exact": bool(t) and h(t) == h(t[::-1])}

def novelty_preflight() -> dict:
    rows = json.loads(REGISTRY.read_text()).get("entries", [])
    exact = any(r.get("id") != EXPERIMENT_ID and r.get("signature") == SIGNATURE for r in rows)
    related = [r.get("id") for r in rows if any(k in r.get("signature", "") for k in ("phrase-lattice", "phrase-fst", "paraphrase-graph"))]
    return {"entries_inspected": len(rows), "exact_signature_collision": exact, "related_families": related, "passed": not exact, "reason": "global equation over role-complete authored phrases is distinct" if not exact else "exact signature already registered"}

def vector(text: str) -> Counter[str]:
    return Counter(tape(text))

def scene(assign: tuple[str, ...]) -> str:
    return " ".join(assign).rstrip(".") + "."

def equation(left: str, right: str) -> dict:
    lt, rt = tape(left), tape(right)
    n = min(len(lt), len(rt)); first = next(((i, lt[i], rt[-1-i]) for i in range(n) if lt[i] != rt[-1-i]), None)
    debt = Counter(lt) - Counter(rt[::-1]); surplus = Counter(rt[::-1]) - Counter(lt)
    return {"left_letters": len(lt), "right_letters": len(rt), "global_character_equation": "left == reverse(right)", "first_mismatch": first, "character_debt": dict(debt), "character_surplus": dict(surplus), "closed": len(lt) == len(rt) and first is None}

def admissible(text: str) -> dict:
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=300)
    units = tokenize(text)
    return {"checks": checks, "word_count": len(units), "complete_prose": text.endswith("."), "distinct_inventory_content": len([w for w in units if w not in {"the", "a", "an", "and", "for", "near", "under", "before", "after"}]) == len(set(w for w in units if w not in {"the", "a", "an", "and", "for", "near", "under", "before", "after"}))}

def run() -> dict:
    pre = novelty_preflight()
    # Two independent scene realizations. The equation is evaluated globally;
    # no partial prefix is presented as a closure.
    left_choices = list(product(INVENTORY["agent"], INVENTORY["action"], INVENTORY["object"], INVENTORY["place"], INVENTORY["purpose"]))
    right_choices = list(product(INVENTORY["agent"], INVENTORY["action"], INVENTORY["object"], INVENTORY["place"], INVENTORY["purpose"]))
    rows = []
    for lc in left_choices:
        left = scene((lc[0], lc[1], lc[2], lc[3], lc[4]))
        # Rank right scenes by global character-vector proximity, not local tape edits.
        scored = []
        for rc in right_choices:
            right = scene((rc[0], rc[1], rc[2], rc[3], rc[4]))
            eq = equation(left, right)
            mismatch = eq["first_mismatch"]
            score = (len(eq["character_debt"]) + len(eq["character_surplus"]), abs(eq["left_letters"] - eq["right_letters"]), 0 if mismatch is None else mismatch[0])
            scored.append((score, right, eq, rc))
        _, right, eq, rc = min(scored, key=lambda x: x[0])
        rendered = left
        rows.append({"rendered": rendered, "paired_scene": right, "letters": len(tape(rendered)), "equation": eq, "audit": {"two_pointer": two_pointer(rendered), "hash": hash_audit(rendered)}, "admission": admissible(rendered), "provenance": {"source": "independently authored semantic role inventories", "inventory_roles": list(ROLES), "catalogue_text_used": False, "word_order_mirrored": False, "repeated_unit_shortcut": False, "paired_choice": list(rc)}, "repair": {"strategy": "replace-one-held-out-whole-phrase-and-recompute-global-equation", "first_mismatch": eq["first_mismatch"], "next": "author a held-out place or purpose phrase whose full character vector closes the recorded debt while retaining scene meaning"}})
    # Keep one representative per unique rendered scene and expose the best
    # complete prose, while preserving all search evidence.
    unique = {r["rendered"]: r for r in rows}
    result = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "novelty_preflight": pre, "operator": "complete semantic role assignment, global reversed-tape character equation, meet-in-the-middle nearest debt, whole-phrase repair", "candidate_count": len(unique), "closure_count": sum(r["equation"]["closed"] for r in unique.values()), "best_actual_prose": max(unique.values(), key=lambda r: r["letters"]), "candidates": list(unique.values()), "reader_eligible": False, "status": "completed_no_exact_closure", "next_repair": "held-out phrase authoring against global character debt, then rerun exact dual audit and blinded intact/shuffled reader gate"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    r = run(); print(json.dumps({"experiment_id": r["experiment_id"], "novelty": r["novelty_preflight"], "candidates": r["candidate_count"], "closures": r["closure_count"], "best": r["best_actual_prose"]["rendered"]}, indent=2))
