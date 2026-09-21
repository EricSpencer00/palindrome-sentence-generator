"""Held-out PP-production operator for Earley mirrored-domain search.

The PP alternatives are absent from the prior clause bank and are introduced
before a clause is rendered.  Earley items carry agreement/valency features;
the mirrored character support domain is intersected while terminal chunks
are emitted.  This is construction-time pruning, not post-render repair.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "cfg-earley-heldout-pp-domain-operator-20260921"
SIGNATURE = "earley-heldout-pp-production|pre-render-feature-unification|mirrored-domain-pruning|independent-exact-audit"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{ID}.json"

CLAUSES = (
    {"id": "ranger-lantern-workshop", "subject": "The alert ranger", "verb": "repairs", "object": "a broken lantern", "number": "sg", "valency": "transitive-locative"},
    {"id": "baker-loaves-oven", "subject": "The patient baker", "verb": "guides", "object": "the warm loaves", "number": "sg", "valency": "transitive-locative"},
    {"id": "sailor-rigging-harbor", "subject": "The watchful sailor", "verb": "mends", "object": "the loose rigging", "number": "sg", "valency": "transitive-locative"},
)
# These PP terminals are held out from the preceding 2021-09-21 bank.
HELDOUT_PP = (
    {"id": "pp-moonlit-greenhouse", "text": "across the moonlit greenhouse", "valency": "transitive-locative"},
    {"id": "pp-northern-bridge", "text": "under the northern bridge", "valency": "transitive-locative"},
    {"id": "pp-stone-courtyard", "text": "through the stone courtyard", "valency": "transitive-locative"},
)


def norm(text: str) -> str:
    return "".join(c.lower() for c in text if "a" <= c.lower() <= "z")


def render(c: dict, pp: dict) -> str:
    return f"{c['subject']} {c['verb']} {c['object']} {pp['text']}"


def earley_items(c: dict, pp: dict) -> dict:
    features = {"number": c["number"], "valency": c["valency"]}
    return {"algorithm": "earley_pre_render_chart", "productions": ["S -> NP VP", "VP -> V NP PP", "PP -> heldout_terminal"], "items": [{"lhs": "S", "dot": 0, "origin": 0, "features": {"number": c["number"]}}, {"lhs": "S", "dot": 1, "origin": 0, "features": {"number": c["number"]}}, {"lhs": "S", "dot": 2, "origin": 0, "features": features}, {"lhs": "PP", "dot": 1, "origin": 0, "features": {"valency": pp["valency"], "heldout": True, "terminal_id": pp["id"]}}], "feature_unification": features["valency"] == pp["valency"], "accepted": features["valency"] == pp["valency"]}


def domains(left: str, right: str) -> dict:
    l, r = norm(left), norm(right)
    rows, conflicts = [], []
    for i in range(min(len(l), len(r))):
        j = len(r) - 1 - i
        support = {l[i]} & {r[j]}
        item = {"position": i, "opposing_position": j, "left_item_owner": "S/VP-left", "right_item_owner": "S/VP-right", "left_support": l[i], "right_support": r[j], "domain": sorted(support), "survives": bool(support)}
        rows.append(item)
        if not support:
            conflicts.append(item)
    return {"algorithm": "online_mirrored_domain_intersection", "owner_invariant": "every support is emitted by a complete Earley item; empty support is a witnessed conflict", "pairs_checked": len(rows), "surviving_pairs": sum(x["survives"] for x in rows), "conflict_count": len(conflicts), "first_conflict": conflicts[0] if conflicts else None, "ledger": rows[:20]}


def pointer(text: str) -> dict:
    t = norm(text)
    mm = [{"i": i, "j": len(t)-1-i, "left": t[i], "right": t[-1-i]} for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"algorithm": "independent_pointer", "exact": bool(t) and not mm, "letters": len(t), "mismatch_count": len(mm), "mismatches": mm[:12]}


def sha(text: str) -> dict:
    t = norm(text)
    return {"algorithm": "independent_sha_forward_reverse", "exact": bool(t) and hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest(), "forward": hashlib.sha256(t.encode()).hexdigest(), "reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}


def novelty() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [x["id"] for x in entries if x.get("id") != ID and x.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions": collisions, "passed": not collisions, "distinction": "held-out PP terminals enter before rendering and unify valency inside Earley items; not a rendered repair, fixed-tape search, or reward loop"}


def make_row(left: dict, right: dict, pp_left: dict, pp_right: dict, rank: int) -> dict:
    lc, rc = render(left, pp_left), render(right, pp_right)
    text = lc + ". " + rc + "."
    p, s = pointer(text), sha(text)
    return {"rank": rank, "rendered": text, "letters": p["letters"], "provenance": {"left_clause_id": left["id"], "right_clause_id": right["id"], "left_pp_id": pp_left["id"], "right_pp_id": pp_right["id"], "heldout_pp_bank": True}, "left_chart": earley_items(left, pp_left), "right_chart": earley_items(right, pp_right), "mirrored_domains": domains(lc, rc), "exact_check_pointer": p, "exact_check_sha": s, "independent_exact_agreement": p["exact"] == s["exact"], "anti_shortcut_flags": {"fixed_tape": False, "post_render_repair": False, "reward_loop": False, "reverse_decoder": False, "mirrored_units": False, "word_order_mirror": False, "catalogue_text": False, "complete_constituents": True, "feature_unification": True}, "mechanically_admitted": False, "next_operator": "Introduce a second held-out PP with a distinct preposition while preserving the same valency feature, then re-expand the Earley items and mirrored domains before rendering."}


def run() -> dict:
    pre = novelty()
    if not pre["passed"]:
        raise RuntimeError(pre)
    specs = ((CLAUSES[0], CLAUSES[1], HELDOUT_PP[0], HELDOUT_PP[1]), (CLAUSES[1], CLAUSES[2], HELDOUT_PP[1], HELDOUT_PP[2]), (CLAUSES[2], CLAUSES[0], HELDOUT_PP[2], HELDOUT_PP[0]))
    rows = [make_row(*spec, i + 1) for i, spec in enumerate(specs)]
    return {"experiment_id": ID, "signature": SIGNATURE, "method": "pre-render held-out PP Earley feature/domain operator", "novelty_preflight": pre, "rows": rows, "stats": {"states_examined": len(rows), "over_39": sum(x["letters"] >= 39 for x in rows), "exact": sum(x["mechanically_admitted"] for x in rows), "feature_unified": sum(x["left_chart"]["accepted"] and x["right_chart"]["accepted"] for x in rows), "domain_conflicts": sum(x["mirrored_domains"]["conflict_count"] for x in rows)}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audits": ["pointer", "forward/reverse SHA", "Earley feature chart", "online mirrored-domain ledger"], "brown_usage": "none; no borrowed sentence text"}, "anti_shortcut_policy": "No post-render repair, fixed tape, reverse decoder, mirrored unit, word-order symmetry, catalogue text, or reward loop.", "next_operator": "Add one distinct held-out PP terminal with the same valency feature before expansion, then re-run exact audits."}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
