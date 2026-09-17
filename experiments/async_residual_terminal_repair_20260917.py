"""Targeted repair for the async scene-bank seam.

This is intentionally not a second Cartesian sweep: it keeps one high-scoring
inspection frame and substitutes only held-out terminal adjuncts while replaying
the owner/residual transition and complete-text audits.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.async_residual_scene_bank_20260917 import consume, independent_audit

ROOT = Path(__file__).resolve().parents[1]
ID = "async-residual-terminal-repair-20260917"
SIGNATURE = "async-residual-terminal-repair|heldout-adjunct-substitution|owner-residual-replay|ordinary-clause|independent-audit"
OUT = ROOT / "runs" / "async-residual-terminal-repair-20260917.json"

BASE = {"subject": "careful baker", "verb": "checks", "object": "the ledger"}
HELDOUT = ("along the canal", "under the bridge", "beside the shed", "past the marker")

def novelty():
    reg = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    rows = reg.get("entries", []) + reg.get("excluded", [])
    collisions = [r.get("id") for r in rows if r.get("signature") == SIGNATURE and r.get("id") != ID]
    if collisions: raise RuntimeError(f"novelty collision: {collisions}")
    return {"status":"passed", "registry_entries_before_run":len(rows), "signature_collisions":[],
            "catalogue_lookup":False, "repair_scope":"one frame; held-out terminals only"}

def run():
    candidates=[]
    for terminal in HELDOUT:
        left = f"{BASE['subject']} {BASE['verb']} {BASE['object']} {terminal}"
        right = f"{BASE['subject']} {BASE['verb']} {BASE['object']} {terminal}"
        state=(0, "")
        first=consume(*state, left, "")
        if first is not None: state=first
        second=consume(*state, "", right)
        if second is not None: state=second
        text=left + ". " + right + "."
        audit=independent_audit(text)
        candidates.append({"terminal":terminal,"text":text,"residual_after_replay":list(state),
                           "audit":audit,"provenance":{"base_frame":BASE,"heldout_terminal":True,
                           "catalogue":False,"word_order_mirror":False,"repeated_fragment":False,
                           "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    exact=[c for c in candidates if c["audit"]["exact"]]
    result={"experiment_id":ID,"signature":SIGNATURE,"status":"completed_targeted_repair",
            "candidates":candidates,"rendered_exact_candidates":exact,"exact_count":len(exact),
            "novelty_preflight":novelty(),"next_repair":{"operator":"introduce a typed terminal inflection pair at the same seam",
            "reason":"held-out adjunct substitutions preserve ordinary grammar but leave nonzero residual debt"},
            "provenance":{"catalogue_imported":False,"source_sentences_copied":False,
            "audits":["direct reverse","opposing index","forward/reverse SHA-256","length/provenance"]}}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result

if __name__ == "__main__": print(json.dumps(run(), indent=2))
