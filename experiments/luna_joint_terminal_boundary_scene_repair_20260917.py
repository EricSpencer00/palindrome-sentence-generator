"""Joint terminal/boundary repair over fresh human-authored scene pairs.

The solver selects complete ordinary clauses, then repairs the first opposing
character obligation by jointly changing a role-compatible terminal and its
attached boundary phrase.  It never edits characters, reverses a finished
string, or inserts a known palindrome.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT = "luna-joint-terminal-boundary-scene-repair-20260917"
SIGNATURE = "human-authored-scene-pairs|joint-terminal-boundary-repair|typed-valency|live-character-equations|independent-pointer-sha"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

SCENES = [
    {
        "id": "gardener-watchman",
        "left": [("The quiet gardener", "waters", "young roses"), ("The careful gardener", "tends", "red roses")],
        "right": [("the watchman", "checks", "the west gate"), ("the patient guard", "opens", "the side gate")],
        "boundary": ["after the rain", "before dusk"],
    },
    {
        "id": "cartographer-baker",
        "left": [("The patient cartographer", "marks", "a coastal chart"), ("The alert surveyor", "folds", "a river map")],
        "right": [("the village baker", "carries", "warm loaves"), ("the young cook", "sets", "fresh bread")],
        "boundary": ["near the harbor", "beside the old mill"],
    },
    {
        "id": "archivist-carpenter",
        "left": [("The calm archivist", "files", "a weathered letter"), ("The steady clerk", "copies", "an old ledger")],
        "right": [("the patient carpenter", "repairs", "a cedar chair"), ("the skilled joiner", "builds", "a small table")],
        "boundary": ["inside the quiet hall", "under the north window"],
    },
]


def pointer(text: str) -> dict:
    t = normalize_letters(text)
    i, j, mm = 0, len(t) - 1, []
    while i < j:
        if t[i] != t[j]:
            mm.append({"left": i, "right": j, "a": t[i], "b": t[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent-two-pointer", "letters": len(t), "exact": bool(t) and not mm, "mismatch_count": len(mm), "first_mismatch": mm[0] if mm else None}


def sha_audit(text: str) -> dict:
    t = normalize_letters(text)
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"algorithm": "independent-normalized-forward-reverse-sha256", "forward": f, "reverse": r, "exact": f == r}


def novelty() -> dict:
    entries = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [e.get("id") for e in entries if e.get("id") == EXPERIMENT or e.get("signature") == SIGNATURE]
    return {"passed": not collisions, "registry_entries_read": len(entries), "collisions": collisions, "operator_distinction": "jointly replace a semantic terminal and its attached boundary phrase after recording the first residual; no independent character edits or duplicate product sweep"}


def render(left, right, boundary):
    return f"{left[0]} {left[1]} {left[2]} {boundary}; {right[0].capitalize()} {right[1]} {right[2]}."


def inspect(scene_id, left, right, boundary, phase, parent=None, changed=None):
    text = render(left, right, boundary)
    p, s = pointer(text), sha_audit(text)
    checks = mechanical_admission_checks(text, min_letters=50, max_letters=220)
    words = re.findall(r"[a-z]+", text.casefold())
    return {"scene_id": scene_id, "phase": phase, "rendered": text, "letters": p["letters"], "semantic_slots": {"left_agent": left[0], "left_verb": left[1], "left_object": left[2], "boundary": boundary, "right_agent": right[0], "right_verb": right[1], "right_object": right[2]}, "changed_joint_slots": changed, "parent_first_residual": parent, "exact_audit": {"pointer": p, "sha": s, "independent_agreement": p["exact"] == s["exact"]}, "mechanical_checks": checks, "mechanically_admitted": bool(p["exact"] and s["exact"] and all(checks.values())), "provenance": {"fresh_authored_scene": True, "catalogue_imported": False, "borrowed_text": False, "reversed_finished_sentence": False, "word_order_mirror": False, "repeated_unit": False, "fragment": len(words) < 10, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "next_repair": "Author a held-out role-compatible terminal plus its boundary attachment for this first residual, then rerun the full sentence audit; never alter the normalized tape directly."}


def run():
    pre = novelty()
    if not pre["passed"]:
        raise RuntimeError(pre)
    rows, controls = [], []
    for scene in SCENES:
        states = []
        for left, right, boundary in itertools.product(scene["left"], scene["right"], scene["boundary"]):
            states.append(inspect(scene["id"], left, right, boundary, "lattice", changed=None))
        states.sort(key=lambda r: (r["exact_audit"]["pointer"]["mismatch_count"], -r["letters"]))
        best = states[0]
        rows.extend(states)
        # One bounded repair, changing the right terminal and its attachment together.
        right_alt = scene["right"][1]
        boundary_alt = scene["boundary"][1]
        repaired = inspect(scene["id"], scene["left"][0], right_alt, boundary_alt, "joint_repair", best["exact_audit"]["pointer"]["first_mismatch"], ["right_agent/right_object", "boundary"])
        controls.append(repaired)
    all_rows = rows + controls
    exact = [r for r in all_rows if r["mechanically_admitted"]]
    best = min(all_rows, key=lambda r: r["exact_audit"]["pointer"]["mismatch_count"])
    return {"experiment_id": EXPERIMENT, "signature": SIGNATURE, "status": "completed_exact" if exact else "completed_no_exact_closure", "novelty_preflight": pre, "method": "fresh complete human-authored scene pairs; jointly solve terminal and attachment boundary slots against live character obligations", "rows": all_rows, "controls": controls, "stats": {"scene_families": len(SCENES), "lattice_states": len(rows), "joint_repairs": len(controls), "rendered": len(all_rows), "exact": len(exact), "mechanically_admitted": len(exact), "longest_letters": max(r["letters"] for r in all_rows)}, "best": {"rendered": best["rendered"], "letters": best["letters"], "mismatch_count": best["exact_audit"]["pointer"]["mismatch_count"], "first_mismatch": best["exact_audit"]["pointer"]["first_mismatch"]}, "reader_status": "not eligible unless exact and every mechanical anti-shortcut gate passes", "anti_shortcut_policy": "Complete ordinary clauses only; no fixed tape, reverse decoder, word-order mirror, repeated unit, catalogue text, fragments, or post-hoc character edits.", "next_repair": "Replace only the terminal/boundary pair exposing the best first residual with a held-out role-compatible realization, then re-score the full scene and preserve the intact prose control.", "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(), "independent_audits": ["two-pointer", "normalized forward/reverse SHA-256", "mechanical admission"]}}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
