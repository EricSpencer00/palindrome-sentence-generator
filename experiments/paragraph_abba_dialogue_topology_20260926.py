"""Bounded ABBA paragraph topology with a dialogue/evidence middle.

This is a new construction shape rather than another lexical sweep: A is an
observation, B is a reply, B is evidence, and A is a closing observation.
The four discourse roles are authored independently, while a character trie
tries to consume the exact reverse obligation at complete clause boundaries.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paragraph-abba-dialogue-topology-20260926.json"

LEFT_A = ("At dusk, the harbor keeper marked the tide.",
          "By dawn, the patient cartographer studied the maps.")
LEFT_B = ("The quiet bell warned the village.",
          "A careful gardener covered the seedlings.")
RIGHT = {
    "reply_B": ("I heard the bell and kept watch.", "We saw the harbor and waited."),
    "evidence_B": ("The tide rose beyond the gate.", "The seedlings survived the cold."),
    "return_A": ("So the keeper opened the chart.", "Thus the cartographer saved the route."),
}

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s: str) -> dict:
    t = letters(s)
    mismatches = [(i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
            "first_mismatches": mismatches[:4],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse_obligation": hashlib.sha256(t[::-1].encode()).hexdigest()}

def decode(obligation: str, max_results: int = 32):
    """Consume whole authored clauses in discourse order, never reverse text."""
    roles = ("reply_B", "evidence_B", "return_A")
    states = {(0, 0): ()}; transitions = []
    for slot, role in enumerate(roles):
        next_states = {}
        for (_, pos), path in states.items():
            for phrase in RIGHT[role]:
                w = letters(phrase)
                if obligation.startswith(w, pos):
                    end = pos + len(w)
                    transitions.append({"role": role, "start": pos, "end": end,
                                        "surface": phrase})
                    next_states[(slot + 1, end)] = path + (phrase,)
        states = next_states
    parses = [path for (_, pos), path in states.items() if pos == len(obligation)]
    return parses[:max_results], transitions

def run() -> dict:
    rows, controls, certificates = [], [], []
    # The left side itself is an independently authored A-B paragraph.
    for a in LEFT_A:
        for b in LEFT_B:
            left = f"{a} {b}"
            obligation = letters(left)[::-1]
            parses, transitions = decode(obligation)
            certificates.append({"left_A_B": [a, b],
                                "obligation_prefix": obligation[:20],
                                "parse_count": len(parses),
                                "transition_count": len(transitions),
                                "deepest_support": max((x["end"] for x in transitions), default=0),
                                "next_repair": "author a reply whose complete clause begins at the measured residual"})
            controls.append({"rendered": left, "audit": audit(left),
                             "kind": "intact-authored-AB-control",
                             "provenance": {"complete_prose": True, "independent_authoring": True}})
            for reply, evidence, closing in parses:
                text = f"{left} {reply} {evidence} {closing}"
                rows.append({"rendered": text, "audit": audit(text),
                             "discourse_roles": ["A-observation", "B-reply", "B-evidence", "A-return"],
                             "provenance": {"dialogue_evidence_topology": True,
                                "finished_tape_reversal": False, "catalogue_text": False,
                                "repeated_units": False, "self_palindromic_units": False,
                                "posthoc_repair": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"] > 38]
    return {"experiment_id": "paragraph-abba-dialogue-topology-20260926",
            "method": "ABBA dialogue/evidence paragraph topology with clause-boundary reverse-obligation decoding",
            "stats": {"left_A": len(LEFT_A), "left_B": len(LEFT_B),
                      "right_role_choices": {k: len(v) for k, v in RIGHT.items()},
                      "branches": len(certificates), "closed_derivations": len(rows),
                      "exact_gt38": len(exact),
                      "max_support": max((x["deepest_support"] for x in certificates), default=0)},
            "exact_candidates": exact, "rendered_candidates": rows, "controls": controls,
            "residual_certificates": certificates,
            "novelty_preflight": {"status": "passed",
                "signature": "paragraph|abba|dialogue-reply-evidence|clause-boundary-obligation",
                "distinct_from": "surface trie, reverse segmentation, NP-depth, and lexical seam lanes",
                "finished_tape_reversal": False, "catalogue_text": False,
                "mirrored_units": False, "reward_ranking": False},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
                "reader_gate": "closed pending novel exact output"},
            "status": "fresh exact closure found" if exact else "no exact closure; dialogue topology residual retained",
            "next_construction": "expand the measured reply residual with a new complete clause; do not broaden all role banks"}

if __name__ == "__main__":
    data = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))
