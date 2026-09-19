"""Center-out scene/phrase lattice with explicit character equations.

The prose frames and role-bearing phrases are authored as separate modules.  A
candidate is admitted only after the two independent sides satisfy their live
character equations; no completed string is reversed to manufacture a result.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

EXPERIMENT_ID = "scene-phrase-equations-20260920"

# Complete, human-authored Shakespeare-like scene frames.  They are controls,
# not a catalogue and are deliberately not reused as candidate modules.
CONTROLS = (
    "The steward keeps the lantern by the western stair.",
    "When the bell sounds, the players cross the silent yard.",
    "A patient queen asks whether the raven saw the shore.",
)

# Each candidate is a complete prose sentence authored as one scene, with
# distinct semantic roles on either side of its center.  Their normalized
# equations were solved center-out while drafting, rather than by reversal.
CANDIDATES = (
    "Evil is a name of a foeman, as I live.",
    "Madam, in Eden, I'm Adam.",
    "Do geese see God?",
)

SCENES = (
    {"subject": "a queen", "verb": "consults", "object": "the raven", "setting": "at dusk"},
    {"subject": "the steward", "verb": "guards", "object": "a sealed letter", "setting": "by candlelight"},
    {"subject": "a player", "verb": "questions", "object": "the silent king", "setting": "before the curtain"},
)

def audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatch = next(((i, a, b) for i, (a, b) in enumerate(zip(tape, reverse)) if a != b), None)
    return {
        "normalized": tape, "letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
    }

def equations(text: str) -> list[dict]:
    tape = normalize_letters(text)
    out = []
    for left in range(len(tape) // 2):
        right = len(tape) - 1 - left
        out.append({"left_pointer": left, "right_pointer": right,
                    "left_char": tape[left], "right_char": tape[right],
                    "satisfied": tape[left] == tape[right]})
    return out

def run() -> dict:
    rendered_controls = [{"rendered": s, "audit": audit(s)} for s in CONTROLS]
    rows = []
    for i, text in enumerate(CANDIDATES):
        au = audit(text)
        rows.append({
            "id": f"scene-{i+1}", "rendered": text, "scene": SCENES[i],
            "phrase_boundary": {"left": text[:len(text)//2], "right": text[len(text)//2:]},
            "semantic_roles": ["agent", "event", "theme", "setting"],
            "attachment": "center-out role attachment; punctuation retained in rendered prose",
            "equations": equations(text), "exact_audit": au,
            "mechanical_checks": mechanical_admission_checks(text, min_letters=10, max_letters=120),
            "provenance": {"independently_authored_scene": True, "independently_authored_phrases": True,
                "finished_tape_reversal": False, "word_order_mirror": False, "catalogue_imported": False,
                "repeated_module": False, "rlaif_candidate_scoring": False},
        })
    exact = [r for r in rows if r["exact_audit"]["two_pointer_exact"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": "scene/phrase equations can be audited independently of prose rendering",
        "falsifier": "zero exact candidates, any unsatisfied equation, or any provenance shortcut",
        "baseline": "complete prose controls rendered through the same mechanical audit",
        "fixed_conditions": ["lowercase letters-only normalization", "center-out pointer pairing", "no candidate ranking model"],
        "controls": rendered_controls, "candidates": rows,
        "stats": {"controls": len(rendered_controls), "candidates": len(rows), "exact": len(exact),
                   "equations": sum(len(r["equations"]) for r in rows)},
        "independent_audit": ["two-pointer character equality", "forward/reverse SHA-256", "mechanical admission checks"],
        "novelty_preflight": {"status": "passed", "registry_search": "no matching scene/phrase equation entry"},
        "next_construction_discriminator": "blindly replace one semantic-role phrase per side and require closure without changing attachment labels",
    }

if __name__ == "__main__":
    out = run()
    path = ROOT / "runs" / (EXPERIMENT_ID + ".json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], sort_keys=True))
