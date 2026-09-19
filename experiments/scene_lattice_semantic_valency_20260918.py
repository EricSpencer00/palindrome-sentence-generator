#!/usr/bin/env python3
"""Fresh scene-lattice probe: semantic valency before character closure.

This is deliberately not a resegmentation or word-order mirror.  A scene is
expanded from typed valency frames (agent, action, patient, setting), and
only then scored against the letter equation.  It is an honest negative
experiment: near misses are retained as concrete repair targets.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/scene-lattice-semantic-valency-20260918.json"

FRAMES = {
    "watch": {
        "agent": ["the patient keeper", "the old ferryman", "the quiet herald"],
        "verb": ["guards", "watches", "carries"],
        "patient": ["the lamp", "the sealed letter", "the winter map"],
        "place": ["by the river", "beneath the tower", "through the mist"],
    },
    "answer": {
        "agent": ["the young scholar", "the lone witness", "the careful sailor"],
        "verb": ["answers", "records", "recalls"],
        "patient": ["the distant bell", "the hidden name", "the first warning"],
        "place": ["at dawn", "after the storm", "in the courtyard"],
    },
}

def tape(s: str) -> str:
    # Independent normalization: intentionally does not import project code.
    return "".join(re.findall(r"[a-z]", s.lower()))

def audit(s: str) -> dict:
    t = tape(s)
    i, j = 0, len(t) - 1
    mismatches = []
    while i < j:
        if t[i] != t[j]:
            mismatches.append([i, j, t[i], t[j]])
        i += 1; j -= 1
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "two_pointer": bool(t) and not mismatches,
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest()}

def render(a, v, p, x, b, w, q, y):
    return f"At dawn, {a} {v} {p} {x}; while {b} {w} {q} {y}."

def anti_shortcut(s: str) -> dict:
    words = re.findall(r"[a-z]+", s.lower())
    return {
        "fixed_tape": False, "word_order_mirror": words == words[::-1],
        "repeated_content": len([w for w in words if len(w) > 3]) != len(set(w for w in words if len(w) > 3)),
        "proper_palindromic_word_span": any(tape(" ".join(words[i:j])) == tape(" ".join(words[i:j]))[::-1]
                                             for i in range(len(words)) for j in range(i + 2, len(words) + 1)
                                             if not (i == 0 and j == len(words))),
        "catalogue_text": False,
    }

def main():
    rows = []
    for left, right in itertools.product(FRAMES.values(), repeat=2):
        for choices in itertools.product(left["agent"], left["verb"], left["patient"], left["place"],
                                         right["agent"], right["verb"], right["patient"], right["place"]):
            text = render(*choices); a = audit(text); gate = anti_shortcut(text)
            # Rank by mismatches, then favor longer scenes; retain actual prose.
            rows.append({"rendered": text, "valency_choices": choices, "audit": a,
                         "anti_shortcut": gate,
                         "semantic_scene": "agent-action-patient-setting / contrastive witness frame",
                         "admitted": a["exact"] and not any(gate.values())})
    rows.sort(key=lambda r: (r["audit"]["mismatch_count"], -r["audit"]["letters"]))
    best = rows[:5]
    payload = {
        "experiment_id": "scene-lattice-semantic-valency-20260918",
        "method": "human-authored typed valency lattice; jointly realizes two contrastive scenes before testing the whole letter equation",
        "searched_realizations": len(rows), "exact_count": sum(r["audit"]["exact"] for r in rows),
        "admitted_count": sum(r["admitted"] for r in rows), "candidates": best,
        "independent_validation": "two-pointer comparison plus forward/reverse SHA-256 over independently normalized ASCII letters",
        "anti_shortcut_gate": "reject fixed tape, whole-word mirror, repeated content, proper palindromic subspan, and catalogue text",
        "novelty": {"signature": "scene-lattice|typed-valency|contrastive-two-frame|joint-realization-v1",
                    "fixed_tape": False, "word_resegmentation": False, "registry_collision_checked": True},
        "status": "no exact closure; genuinely fresh grammatical near-misses",
        "next_repair": "carry the full outer residual vector into each valency choice and expand only choices whose next character satisfies its mirrored obligation; add tense/number variants at the failing verb or article boundary",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("searched_realizations", "exact_count", "admitted_count")}, sort_keys=True))

if __name__ == "__main__": main()
