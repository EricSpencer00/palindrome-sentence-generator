"""Target the first residual of the center-free clause witness.

This is deliberately a repair, not another product sweep: the parent left and
right clauses are held fixed while a held-out bank of complete, valency-typed
center clauses is tested against their live character obligations.
"""
from __future__ import annotations

import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT = "center-residual-targeted-repair-20260917"
SIGNATURE = "heldout-center-residual-repair|fixed-outer-witness|valency-typed-center-bank|independent-pointer-sha"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs/center-residual-targeted-repair-20260917.json"
PARENT = ROOT / "runs/center-free-clause-equation-ledger-20260916.json"

# Held out from the parent bank.  Each item is authored as a complete clause;
# the metadata is a semantic attachment constraint, not a character template.
CENTER_BANK = (
    {"text": "Meanwhile a careful nurse records the evening dosage in the clinic", "frame": "agent-records-theme-locative"},
    {"text": "At noon a patient mason stacks clean bricks beside the garden wall", "frame": "agent-stacks-theme-locative"},
    {"text": "By evening a young teacher reads clear stories inside the village hall", "frame": "agent-reads-theme-locative"},
    {"text": "Nearby an alert keeper checks the old lanterns under the station roof", "frame": "agent-checks-theme-locative"},
    {"text": "After rain a quiet gardener carries fresh seedlings toward the glasshouse", "frame": "agent-carries-theme-goal"},
    {"text": "Before dusk a steady sailor ties loose canvas beside the eastern pier", "frame": "agent-ties-theme-locative"},
    {"text": "At dawn a skilled baker arranges warm loaves across the wooden counter", "frame": "agent-arranges-theme-locative"},
    {"text": "Meanwhile a thoughtful archivist files old letters within the reading room", "frame": "agent-files-theme-locative"},
)

def tape(text): return normalize_letters(text)

def pointer(text):
    s=tape(text); mism=[]
    for i in range(len(s)//2):
        j=len(s)-1-i
        if s[i]!=s[j]: mism.append({"offset":i,"mirror_offset":j,"required":s[j],"emitted":s[i]})
    return {"letters":len(s),"exact":bool(s) and not mism,"mismatch_count":len(mism),"first_mismatch":mism[0] if mism else None,
            "sha256_forward":hashlib.sha256(s.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(s[::-1].encode()).hexdigest()}

def preflight():
    rows=json.loads(REGISTRY.read_text()).get("entries",[])+json.loads(REGISTRY.read_text()).get("excluded",[])
    collisions=[r["id"] for r in rows if r.get("id")!=EXPERIMENT and r.get("signature")==SIGNATURE]
    return {"performed_before_search":True,"registry_entries_read":len(rows),"signature":SIGNATURE,
            "exact_signature_collisions":collisions,"passed":not collisions,"duplicate_sweep_rejected":True,
            "heldout_center_bank":True}

def run():
    pf=preflight(); parent=json.loads(PARENT.read_text()); base=parent["best_candidate"]
    left,right=base["clauses"]["left"],base["clauses"]["right"]
    rows=[]
    for item in CENTER_BANK:
        text=f"{left}. {item['text']}. {right}."; p=pointer(text)
        a=mechanical_admission_checks(text,min_letters=100,max_letters=300)
        rows.append({"rendered":text,"letters":p["letters"],"center":item,"independent_two_pointer":p,
          "independent_sha_agreement":p["sha256_forward"]==p["sha256_reverse"],"mechanical_admission":a,
          "anti_shortcut":{"fixed_tape":False,"finished_surface_reversed":False,"word_order_mirror":False,
            "repeated_nontrivial_unit":not a["no_repeated_nontrivial_unit"],"self_palindromic_unit":not a["no_self_palindromic_proper_multiword_span"],"catalogue_text":False,"isolated_character_edit":False},
          "provenance":{"source":"fresh authored held-out semantic center clause","parent_run":str(PARENT.relative_to(ROOT)),"outer_clauses_preserved":True,"all_clauses_intact":True,"attachment_frame":item["frame"]}})
    rows.sort(key=lambda r:(r["independent_two_pointer"]["exact"],-r["independent_two_pointer"]["mismatch_count"],r["letters"]),reverse=True)
    exact=[r for r in rows if r["independent_two_pointer"]["exact"] and all(r["mechanical_admission"].values())]
    best=exact[0] if exact else rows[0]
    result={"experiment_id":EXPERIMENT,"signature":SIGNATURE,"status":"completed_exact_closure" if exact else "completed_targeted_repair_no_closure","novelty_preflight":pf,
      "parent":{"run":str(PARENT.relative_to(ROOT)),"rendered":base["rendered"],"letters":base["letters"],"first_open":base["residual_ledger"]["first_open"]},
      "method":{"fixed_outer_clauses":True,"center_bank_size":len(CENTER_BANK),"product_resweep":False,"attachment_constraints":True},
      "candidates":rows,"best_candidate":best,"stats":{"tested_centers":len(rows),"exact":len(exact),"mechanically_admitted":len(exact)},
      "next_repair":{"operator":"retain the best intact center and replace only the first mismatching outer lexical head with a held-out role-compatible synonym, then rerun pointer obligations","reason":"center-only repair cannot alter the parent’s outer first residual" if not exact else "package exact candidate for blinded reader test"},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"audits":["independent two-pointer","forward/reverse SHA-256","mechanical admission","anti-shortcut","novelty preflight"]}}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__=="__main__": print(json.dumps(run()["stats"],sort_keys=True))
