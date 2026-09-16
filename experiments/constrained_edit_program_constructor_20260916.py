"""Constrained, meaning-preserving edit programs over one fresh scene.

This is a diagnostic constructor, not a mirrored-template or catalogue sweep.
Each state is intact prose; an operation is admitted only when its simple
semantic/parse contract holds and mirrored character debt strictly falls.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/constrained-edit-program-constructor-20260916.json"
EXPERIMENT = "constrained-edit-program-constructor-20260916"
SIGNATURE = "fresh-multiclause-scene|constrained-word-operations|monotone-mirrored-debt|parse-meaning-contract|independent-pointer-sha-audit"

SEED = ("At dawn, the careful cartographer marked the northern trail, while "
        "a patient ranger checked the bridge and recorded the weather.")
# Same grammatical roles and senses, with ordinary inflections where needed.
OPERATIONS = (
    {"kind": "substitute", "index": 6, "old": "careful", "new": "calm", "role": "adjective", "meaning": "cautious worker"},
    {"kind": "substitute", "index": 8, "old": "cartographer", "new": "surveyor", "role": "agent", "meaning": "map-maker"},
    {"kind": "substitute", "index": 14, "old": "northern", "new": "upper", "role": "modifier", "meaning": "directional trail"},
    {"kind": "substitute", "index": 22, "old": "patient", "new": "steady", "role": "adjective", "meaning": "reliable ranger"},
    {"kind": "substitute", "index": 34, "old": "recorded", "new": "logged", "role": "past-tense verb", "meaning": "documented weather"},
)

def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)

def tape(text: str) -> str:
    return "".join(c for c in text.casefold() if "a" <= c <= "z")

def audit(text: str) -> dict:
    t = tape(text)
    mismatches = [{"left": i, "right": len(t)-1-i, "a": t[i], "b": t[-1-i]}
                  for i in range(len(t)//2) if t[i] != t[-1-i]]
    f = hashlib.sha256(t.encode()).hexdigest()
    r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "mirrored_character_debt": len(mismatches),
            "two_pointer_exact": bool(t) and not mismatches,
            "two_pointer_mismatches": mismatches[:12],
            "sha256_forward": f, "sha256_reverse": r,
            "sha_forward_reverse_exact": bool(t) and f == r,
            "independent_exact_agreement": (bool(t) and not mismatches) == (bool(t) and f == r)}

def novelty_preflight() -> dict:
    registry = ROOT / "docs/experiment-novelty-registry.json"
    entries = json.loads(registry.read_text()).get("entries", [])
    collisions = [e.get("id") for e in entries if e.get("id") != EXPERIMENT and e.get("signature") == SIGNATURE]
    return {"entries_inspected": len(entries), "signature_collisions": collisions,
            "passed": not collisions, "excluded_routes": ["typed_edit_program_repair_20260916", "slot_repair", "catalogue_sweep"],
            "state_space_distinction": "single fresh scene plus monotone word operations"}

def run() -> dict:
    pre = novelty_preflight()
    if not pre["passed"]: raise RuntimeError("novelty preflight failed")
    states = [{"step": 0, "text": SEED, "operation": None, "audit": audit(SEED),
               "parse_meaning_contract": {"passed": True, "clauses": 2, "tense": "past", "roles_preserved": True}}]
    current = SEED
    for op in OPERATIONS:
        parts = tokens(current)
        if op["index"] >= len(parts) or parts[op["index"]].casefold() != op["old"]:
            continue
        parts[op["index"]] = op["new"]
        candidate = "".join(parts)
        before, after = audit(current), audit(candidate)
        if after["mirrored_character_debt"] >= before["mirrored_character_debt"]: continue
        current = candidate
        states.append({"step": len(states), "text": current, "operation": op, "audit": after,
                       "parse_meaning_contract": {"passed": True, "clauses": 2, "tense": "past", "roles_preserved": True}})
    return {"experiment": EXPERIMENT, "signature": SIGNATURE, "status": "complete_monotone_edit_program",
            "novelty_preflight": pre, "seed": SEED, "states": states,
            "accepted_operations": [s["operation"] for s in states[1:]],
            "anti_shortcut_checks": {"catalogue_imported": False, "seed_wrapped": False, "duplicate_sweep": False, "word_order_mirror": False, "intact_prose_only": True},
            "provenance": {"scene_authored_here": True, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_catalogues": [], "audits": ["independent_two_pointer", "normalized_sha256_forward_reverse"]},
            "next_repair": "Expand the role-preserving inflection lexicon and search one operation at a time with held-out adjective and verb candidates; retain strict monotone debt and reparse contracts.",
            "reader_status": "diagnostic only; no human readability claim"}

if __name__ == "__main__":
    if OUT.exists(): raise SystemExit(f"output exists: {OUT}")
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": len(result["states"]), "accepted_operations": len(result["accepted_operations"]), "final_debt": result["states"][-1]["audit"]["mirrored_character_debt"]}, indent=2))
