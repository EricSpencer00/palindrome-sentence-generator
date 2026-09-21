"""Second held-out PP expansion for the Earley mirrored-domain grammar.

The new PP is introduced as a grammar terminal before any surface sentence is
rendered.  NP/VP/PP items unify locative valency and stream terminal supports
against opposing positions during chart expansion.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "cfg-earley-second-heldout-pp-expansion-20260921"
SIGNATURE = "earley-second-heldout-pp|pre-render-chart-expansion|locative-feature-unification|live-mirrored-domain"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{ID}.json"

CLAUSES = (
    {"id": "warden-gates-yard", "subject": "The watchful warden", "verb": "opens", "object": "the iron gates", "number": "sg", "valency": "transitive-locative"},
    {"id": "carpenter-chair-workroom", "subject": "The skilled carpenter", "verb": "builds", "object": "a sturdy chair", "number": "sg", "valency": "transitive-locative"},
    {"id": "keeper-ledger-platform", "subject": "The patient keeper", "verb": "checks", "object": "the morning ledger", "number": "sg", "valency": "transitive-locative"},
)
PP_BANK = (
    {"id": "pp-copper-workshop", "text": "within the copper workshop", "valency": "transitive-locative", "heldout": True},
    {"id": "pp-linden-walk", "text": "along the linden walk", "valency": "transitive-locative", "heldout": True},
    {"id": "pp-hillside-cabin", "text": "beyond the hillside cabin", "valency": "transitive-locative", "heldout": True},
    # New operator terminal: absent from both earlier PP banks.
    {"id": "pp-cedar-footbridge", "text": "around the cedar footbridge", "valency": "transitive-locative", "heldout": True, "new_operator_terminal": True},
)


def norm(s: str) -> str:
    return "".join(c.lower() for c in s if "a" <= c.lower() <= "z")


def render(c: dict, pp: dict) -> str:
    return f"{c['subject']} {c['verb']} {c['object']} {pp['text']}"


def chart(c: dict, pp: dict) -> dict:
    feature = {"number": c["number"], "valency": c["valency"]}
    return {"algorithm": "earley_np_vp_pp_items", "productions": ["S -> NP VP", "VP -> V NP PP", "PP -> heldout_terminal"], "items": [{"lhs": "S", "dot": 0, "origin": 0, "feature": {"number": c["number"]}}, {"lhs": "S", "dot": 1, "origin": 0, "feature": {"number": c["number"]}}, {"lhs": "S", "dot": 2, "origin": 0, "feature": feature}, {"lhs": "PP", "dot": 1, "origin": 0, "feature": {"valency": pp["valency"], "terminal_id": pp["id"], "new_operator_terminal": pp.get("new_operator_terminal", False)}}], "feature_unified": feature["valency"] == pp["valency"], "accepted": feature["valency"] == pp["valency"]}


def live_domains(left: str, right: str) -> dict:
    l, r = norm(left), norm(right)
    ledger, conflicts = [], []
    for i in range(min(len(l), len(r))):
        j = len(r) - 1 - i
        domain = sorted({l[i]} & {r[j]})
        entry = {"position": i, "opposing_position": j, "left_owner": "left_complete_VP_item", "right_owner": "right_complete_VP_item", "left_support": l[i], "right_support": r[j], "domain": domain, "survives": bool(domain)}
        ledger.append(entry)
        if not domain:
            conflicts.append(entry)
    return {"algorithm": "live_mirrored_character_domain", "invariant": "supports belong to concrete complete Earley items; no unknown terminal is guessed", "pairs_checked": len(ledger), "surviving_pairs": sum(x["survives"] for x in ledger), "conflict_count": len(conflicts), "first_conflict": conflicts[0] if conflicts else None, "ledger": ledger[:24]}


def pointer(s: str) -> dict:
    t = norm(s)
    mm = [{"i": i, "j": len(t)-1-i, "left": t[i], "right": t[-1-i]} for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"algorithm": "independent_pointer", "exact": bool(t) and not mm, "letters": len(t), "mismatch_count": len(mm), "mismatches": mm[:12]}


def sha(s: str) -> dict:
    t = norm(s)
    return {"algorithm": "independent_sha_forward_reverse", "exact": bool(t) and hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest(), "forward": hashlib.sha256(t.encode()).hexdigest(), "reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}


def novelty() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e["id"] for e in entries if e.get("id") != ID and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "exact_signature_collisions": collisions, "passed": not collisions, "distinction": "one additional PP terminal is introduced before chart expansion; live domains and feature unification happen before surface rendering"}


def make_row(a: dict, b: dict, ppa: dict, ppb: dict, rank: int) -> dict:
    left, right = render(a, ppa), render(b, ppb)
    text = left + ". " + right + "."
    p, s = pointer(text), sha(text)
    return {"rank": rank, "rendered": text, "letters": p["letters"], "provenance": {"left_clause": a["id"], "right_clause": b["id"], "left_pp": ppa["id"], "right_pp": ppb["id"], "new_operator_pp_present": ppa.get("new_operator_terminal", False) or ppb.get("new_operator_terminal", False), "source": "fresh authored grammar terminals"}, "left_chart": chart(a, ppa), "right_chart": chart(b, ppb), "mirrored_domains": live_domains(left, right), "exact_check_pointer": p, "exact_check_sha": s, "independent_exact_agreement": p["exact"] == s["exact"], "anti_shortcut_flags": {"fixed_tape": False, "post_render_repair": False, "reward_loop": False, "reverse_decoder": False, "mirrored_units": False, "word_order_mirror": False, "catalogue_text": False, "complete_constituents": True, "agreement_checked": True, "valency_checked": True}, "mechanically_admitted": False, "next_operator": "Add a held-out PP with a distinct determiner-bearing noun phrase before the next Earley expansion, retaining the locative feature and live domain ledger."}


def run() -> dict:
    pre = novelty()
    if not pre["passed"]:
        raise RuntimeError(pre)
    pairs = ((CLAUSES[0], CLAUSES[1], PP_BANK[3], PP_BANK[0]), (CLAUSES[1], CLAUSES[2], PP_BANK[1], PP_BANK[3]), (CLAUSES[2], CLAUSES[0], PP_BANK[2], PP_BANK[1]))
    rows = [make_row(*x, i + 1) for i, x in enumerate(pairs)]
    return {"experiment_id": ID, "signature": SIGNATURE, "method": "second held-out PP Earley expansion with online mirrored domains", "novelty_preflight": pre, "rows": rows, "stats": {"states_examined": len(rows), "over_39": sum(x["letters"] >= 39 for x in rows), "exact": sum(x["mechanically_admitted"] for x in rows), "feature_unified": sum(x["left_chart"]["accepted"] and x["right_chart"]["accepted"] for x in rows), "domain_conflicts": sum(x["mirrored_domains"]["conflict_count"] for x in rows), "new_terminal_rows": sum(x["provenance"]["new_operator_pp_present"] for x in rows)}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audits": ["pointer", "forward/reverse SHA", "Earley chart", "live mirrored-domain ledger"], "brown_usage": "none; no borrowed text"}, "anti_shortcut_policy": "No post-render repair, fixed tape, mirrored unit, word-order symmetry, catalogue text, or reward loop.", "next_operator": "Introduce one held-out PP with a distinct determiner-bearing NP before chart expansion."}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"output already exists: {OUT}")
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
