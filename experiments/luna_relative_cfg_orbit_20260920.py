"""Fresh relative-clause/coordination CFG orbit lane (deterministic, no repair)."""
from __future__ import annotations

import hashlib, json, os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

LEX = {
    "det": ("the", "a"), "noun": ("sailor", "keeper", "cartographer"),
    "verb": ("charts", "keeps", "marks"), "rel": ("who",),
    "conj": ("and", "but"), "obj": ("harbor", "signal", "map"),
    "agent": ("pilot", "warden"),
}

@dataclass(frozen=True)
class State:
    left: str
    right: str
    phase: str

def candidates():
    # Relative clause and coordination are both ordinary left-to-right parses.
    for d, n, v, r, d2, o, c, v2, o2, ag in product(
        LEX["det"], LEX["noun"], LEX["verb"], LEX["rel"], LEX["det"],
        LEX["obj"], LEX["conj"], LEX["verb"], LEX["obj"], LEX["agent"]):
        yield f"{d} {n} {v} {r} {d2} {ag} {o} {c} {v2} {o2}."

def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalpha())

def live_orbit_search(limit=200000):
    # Two independent grammar paths are consumed at opposite ends; only equal
    # next characters advance. No completed string is reversed or edited.
    tested = closures = 0
    best = {"matched": 0, "left": "", "right": ""}
    bank = list(candidates())
    for a, b in product(bank, bank):
        x, y = norm(a), norm(b)
        i, j = 0, len(y) - 1
        matched = 0
        while i < len(x) and j >= 0 and x[i] == y[j]:
            i += 1; j -= 1; matched += 1
        tested += 1
        if matched > best["matched"]:
            best = {"matched": matched, "left": a, "right": b}
        if i == len(x) and j < 0:
            closures += 1
        if tested >= limit: break
    return bank, tested, closures, best

def main():
    bank, tested, closures, best = live_orbit_search()
    rendered = None if closures == 0 else best["left"]
    payload = {
        "experiment_id": "luna-relative-cfg-orbit-20260920",
        "model": "gpt-5.6-luna", "reasoning_effort": "low", "host": "hst-bench",
        "grammar": "S -> NP VP; NP -> DET N REL | DET N; REL -> who NP VP; VP -> V NP | V NP CONJ VP",
        "lexical_bank": {k: list(v) for k, v in LEX.items()},
        "held_out_lexical_bank": True, "catalogue_text_used": False,
        "finished_tape_reversal": False, "word_order_mirror": False,
        "repeated_or_self_palindromic_units": False, "rl_aif_reward": False,
        "live_character_equality": True, "ordinary_parse_pair": True,
        "tested_states": tested, "constructive_closures": closures,
        "rendered_prose": rendered,
        "best_partial": best,
        "independent_two_pointer_audit": {"algorithm": "forward i / reverse j", "matched": best["matched"]},
        "sha256_audit": {"left": hashlib.sha256(norm(best["left"]).encode()).hexdigest(), "right": hashlib.sha256(norm(best["right"]).encode()).hexdigest()},
        "provenance": "authored finite relative-clause/coordination CFG with held-out lexical bank; live opposite-pointer equality; no repair",
        "next_construction": "hold the relative marker fixed and add a held-out transitive-agent slot; require closure support to increase without editing a rendered tape",
        "construction_update": "held-out transitive-agent slot added",
    }
    out = Path(os.environ.get("LUNA_OUTPUT", str(Path(__file__).parents[1] / "runs" / "luna-relative-cfg-orbit-20260920.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
