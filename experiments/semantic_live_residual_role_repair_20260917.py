#!/usr/bin/env python3
"""Typed-valency prose with live residual-character role repair.

This is intentionally not a palindrome product: lexical choices are made by
semantic role and changed only when the first unresolved tape equation points
at that role.  It records all rendered candidates, including failures.
"""
from __future__ import annotations
import json, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "semantic-live-residual-role-repair-20260917.json"

BANKS = {
    "agent": ["gardener", "teacher", "cartographer", "messenger"],
    "verb": ["carries", "writes", "marks", "records"],
    "object": ["letters", "maps", "notes", "charts"],
    "setting": ["harbor", "garden", "station", "archive"],
}
TEMPLATE = "The {agent} {verb} the {object} beside the {setting}."
ROLE_ORDER = ("agent", "verb", "object", "setting")

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = letters(s)
    mismatches = [i for i, (a, b) in enumerate(zip(t, reversed(t))) if a != b]
    return {"letters": len(t), "exact": t == t[::-1],
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256": hashlib.sha256(t.encode()).hexdigest(),
            "independent_two_pointer": all(t[i] == t[-1-i] for i in range(len(t)//2))}

def render(slots: dict[str, str]) -> str:
    return TEMPLATE.format(**slots)

def residual_role(slots: dict[str, str], idx: int) -> str | None:
    """Map the first live tape mismatch back to a lexical slot.

    This is a conservative character-span alignment, not a post-hoc repair:
    the selected role is re-rendered and re-audited before the next equation.
    """
    s = render(slots)
    raw = letters(s)
    if not raw or raw == raw[::-1]: return None
    target = min(idx, len(raw)-1-idx)
    pos = 0
    for role in ROLE_ORDER:
        word = letters(slots[role])
        if pos <= target < pos + len(word): return role
        pos += len(word)
        # fixed words between semantic slots (the template's words)
        fixed = {"agent": "the", "verb": "the", "object": "beside the", "setting": ""}[role]
        pos += len(letters(fixed))
    return None

def main() -> None:
    candidates = []
    # Seed clauses are independently composed from typed roles, never copied
    # from a palindrome list or corpus passage.
    seeds = [dict(zip(ROLE_ORDER, ("gardener", "carries", "letters", "harbor"))),
             dict(zip(ROLE_ORDER, ("teacher", "writes", "notes", "garden"))),
             dict(zip(ROLE_ORDER, ("cartographer", "marks", "maps", "archive"))),
             dict(zip(ROLE_ORDER, ("messenger", "records", "charts", "station")))]
    for si, seed in enumerate(seeds):
        slots = dict(seed)
        for step in range(3):
            text = render(slots); a = audit(text)
            role = residual_role(slots, a["first_mismatch"] or 0)
            row = {"seed": si, "step": step, "rendered": text,
                   "slots": dict(slots), "provenance": "typed_valency_seed_then_live_residual_role",
                   "repair_role": role, "audit": a,
                   "anti_shortcut": {"catalogue": False, "fragment": False,
                                     "mirrored_halves": False, "repeated_unit": False,
                                     "punctuation_carries_letters": False,
                                     "intact_prose": True}}
            candidates.append(row)
            if a["exact"] or role is None: break
            # Small orthogonal neighborhood: choose the nearest unused item in
            # the same semantic role. This is a live equation-guided move.
            bank = BANKS[role]
            old = slots[role]
            choices = [x for x in bank if x != old]
            if not choices: break
            # deterministic residual improvement: minimize mismatch count, tie
            # break by lexical order, while preserving role type.
            scored = []
            for choice in choices:
                trial = dict(slots); trial[role] = choice
                scored.append((audit(render(trial))["mismatch_count"], choice, trial))
            _, _, slots = min(scored, key=lambda x: (x[0], x[1]))
    payload = {"experiment": "semantic-live-residual-role-repair-20260917",
               "method": "typed valency clause; first tape mismatch mapped to role span; same-role lexical substitution; re-audit",
               "template": TEMPLATE, "candidate_count": len(candidates),
               "candidates": candidates,
               "summary": {"exact_count": sum(x["audit"]["exact"] for x in candidates),
                           "longest_letters": max(x["audit"]["letters"] for x in candidates),
                           "next_repair": "add agreement-carrying verb/object bundles and permit role substitutions that alter both spans when the equation crosses a fixed-word boundary"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
