"""Bounded repair for the dialogue speech-act residual route.

This is not a new family: it reuses the parent grammar and changes only one
content lexical slot per side with held-out same-act alternatives, replaying
the residual from the first changed character.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dialogue_speech_act_residual_20260916 import (
    EXPERIMENT_ID as PARENT, LEFT, RIGHT, PAIRS, audit, complete, residual,
)

EXPERIMENT_ID = "dialogue-speech-act-residual-repair-20260916"
OUT = ROOT / "runs" / (EXPERIMENT_ID + ".json")
HELD_OUT = {
    "request": {"send": ("share", "deliver"), "bring": ("carry", "fetch")},
    "greeting": {"hello": ("goodbye", "welcome")},
    "report": {"sent": ("shared", "filed"), "found": ("located", "saved")},
    "answer": {"send": ("share", "deliver"), "bring": ("carry", "fetch")},
    "acknowledgment": {"hello": ("welcome", "greetings")},
    "response": {"arrived": ("came", "reached"), "found": ("located", "saved")},
}

def variants(act: str, text: str):
    words = text.split()
    for i, word in enumerate(words):
        for replacement in HELD_OUT.get(act, {}).get(word, ()):
            candidate = words[:]
            candidate[i] = replacement
            yield " ".join(candidate), i, word, replacement

def run():
    rows, stats = [], {"base_pairs": 0, "repair_trials": 0, "complete_trials": 0, "exact_over_38": 0, "max_letters": 0}
    for left in LEFT:
        for right in RIGHT:
            if right.act not in PAIRS.get(left.act, set()): continue
            stats["base_pairs"] += 1
            for ltext, li, old_l, new_l in variants(left.act, left.text):
                for rtext, ri, old_r, new_r in variants(right.act, right.text):
                    stats["repair_trials"] += 1
                    if not complete(ltext + ".") or not complete(rtext + "."): continue
                    stats["complete_trials"] += 1
                    exact, trace, debt = residual(ltext, rtext)
                    rendered = ltext.capitalize() + ". " + rtext.capitalize() + "."
                    row = audit(type(left)(left.act, ltext), type(right)(right.act, rtext))
                    row.update({"changed_left": {"slot": li, "from": old_l, "to": new_l}, "changed_right": {"slot": ri, "from": old_r, "to": new_r}, "replayed_trace": trace, "first_residual_debt": debt, "parent_experiment": PARENT})
                    rows.append(row); stats["max_letters"] = max(stats["max_letters"], row["letters"])
                    stats["exact_over_38"] += int(row["exact"] and row["letters"] > 38)
    return {"experiment_id": EXPERIMENT_ID, "parent_experiment": PARENT,
            "status": "complete_repair_run", "repair_operator": "speech-act-preserving-lexical-substitution-from-first-residual",
            "held_out_same_act_alternatives": HELD_OUT, "stats": stats, "rows": rows,
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_text_imported": False, "complete_utterances_only": True, "new_family": False}}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(), indent=2) + "\n"); print(json.dumps(run()["stats"], sort_keys=True))
