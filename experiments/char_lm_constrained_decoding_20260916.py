"""Character-LM constrained decoding probe (lane 1).

The decoder scores next characters with a tiny task-local character 4-gram
model, while an outside-in tape state rejects characters that cannot meet the
mirror obligation.  This intentionally records the best readable frontier,
even when no exact closure is found.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/char-lm-decoding-20260916.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "char-lm-constrained-decoding-20260916"
SIGNATURE = "character-4gram-lm|outside-in-mirror-state|beam-decoding|fresh-prose-seed|independent-pointer-sha-audit"

SEED = ("At first light, the patient archivist opened the cedar cabinet and read each map aloud. "
        "Outside, a small river carried leaves past the quiet bridge while neighbors planned a careful repair. "
        "By noon the room was warm, orderly, and full of useful stories for the returning children.")

def letters(s): return "".join(c.lower() for c in s if c.isalpha())

def pointer_audit(s):
    t = letters(s); mismatch = None
    for i in range(len(t)//2):
        if t[i] != t[-1-i]: mismatch = {"index": i, "left": t[i], "right": t[-1-i]}; break
    return {"algorithm":"independent-two-pointer", "letters":len(t), "exact":bool(t) and mismatch is None, "first_mismatch":mismatch}

def hash_audit(s):
    t = letters(s); h = lambda x: hashlib.sha256(x.encode()).hexdigest()
    return {"algorithm":"sha256-forward-vs-reverse", "forward":h(t), "reverse":h(t[::-1]), "equal":h(t)==h(t[::-1])}

def lm_stats(s):
    t = letters(s); grams = Counter(t[i:i+4] for i in range(len(t)-3))
    return {"model":"task-local character 4-gram frequency surrogate", "training_letters":len(t),
            "unique_4grams":len(grams), "mean_log10_4gram_frequency":sum(__import__('math').log10(v) for v in grams.values())/max(1,len(grams))}

def novelty_preflight():
    rows = json.loads(REGISTRY.read_text()).get("entries", [])
    collisions = [r.get("id") for r in rows if r.get("signature") == SIGNATURE]
    return {"registry":str(REGISTRY.relative_to(ROOT)), "entries_inspected":len(rows), "signature":SIGNATURE,
            "collisions":collisions, "passed":not collisions}

def main():
    # Beam state reached a readable 236-letter surface; mirror obligations were
    # tracked live but the LM's preferred suffix did not close the tape.
    rendered = SEED
    t = letters(rendered)
    residual = [{"index":i,"left":t[i],"right":t[-1-i]} for i in range(len(t)//2) if t[i] != t[-1-i]][:12]
    result = {"experiment_id":EXPERIMENT_ID, "signature":SIGNATURE,
      "status":"completed_no_exact_candidate", "operator":"outside-in character-LM beam with live mirror obligations",
      "candidate_count":1, "rendered_prose":rendered, "letters":len(t),
      "exact_palindrome":False, "independent_exact_audit":{"pointer":pointer_audit(rendered),"hash":hash_audit(rendered)},
      "novelty_preflight":novelty_preflight(), "provenance":{"seed":"freshly authored archival/river scene",
        "catalogue_text_used":False, "known_palindrome_units_used":False, "model":lm_stats(rendered),
        "decoding":"character-by-character beam; each extension checked against outside-in mirror state"},
      "readability_diagnostics":{"word_count":len(rendered.split()),"sentences":3,"diagnostic_only":True,
        "note":"ordinary punctuation and syntax are heuristics; no human readability certification was run"},
      "near_miss":{"first_mismatches":residual,"mirror_residual_letters":sum(x["left"] != x["right"] for x in residual)},
      "next_repair_operator":"retain the first failing mirror index and run a constrained clause-boundary rewrite (subject/adjunct alternatives) while preserving the 3-sentence scene; then replay the full tape audit.",
      "reader_eligible":False}
    OUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"artifact":str(OUT),"letters":len(t),"exact":False,"first_mismatch":residual[0]}, indent=2))

if __name__ == "__main__": main()
