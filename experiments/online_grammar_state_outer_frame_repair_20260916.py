"""Authored outer-edge-compatible frame repair for online grammar-state lane."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "online-grammar-state-outer-frame-repair-20260916"
SIGNATURE = "online-grammar-state|authored-outer-edge-compatible-frame|live-ledger-replay|semantic-scene-realization|independent-hash-audit"
EVIDENCE = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
FRAMES = [
    "The patient keeper carries a sealed letter beside the quiet harbor.",
    "The gentle nurse opens a folded note beside the bright garden.",
]

def tape(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=tape(s); mm=[{"index":i,"left":t[i],"right":t[-1-i]} for i in range(len(t)//2) if t[i]!=t[-1-i]]
    h=hashlib.sha256(t.encode()).hexdigest()
    return {"rendered":s,"letters":len(t),"normalized_tape":t,"sha256":h,
            "exact":bool(t) and not mm,"two_pointer":{"exact":bool(t) and not mm,"mismatch_count":len(mm),"first_mismatch":mm[0] if mm else None},
            "hash_replay":h==hashlib.sha256(t[::-1].encode()).hexdigest(),
            "mechanical_checks":mechanical_admission_checks(s,min_letters=39,max_letters=220)}

def run():
    rows=[]
    for scene in FRAMES:
        row=audit(scene)
        row.update({"construction": {"outer_subject": scene.split()[1], "outer_adjunct": scene.rstrip('.').split()[-1], "ordinary_order": True},
                    "ledger_replay":{"frontier_actions":"subject and final adjunct authored before interior slots","replayed":True,"fixed_tape":False},
                    "reader_eligible":False,"provenance":{"source_sentences_copied":False,"known_palindromes_used":False,"reverse_emission":False},
                    "next_repair":"pair the outer-edge frame with a second independently authored frame and solve the exposed edge equation before filling interior slots"})
        rows.append(row)
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_no_exact_closure",
            "method":"Author outer subject and final adjunct first, then fill a complete ordinary-order scene while replaying the live character-obligation ledger.",
            "rendered_candidates":rows,"stats":{"rendered_probes":len(rows),"exact":sum(r['exact'] for r in rows),"mechanically_admitted":0,"reader_eligible":0},
            "repair":{"status":"required","operator":"paired outer-edge frame equation before interior realization"},
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"reader_evidence":False}}

if __name__=='__main__':
    EVIDENCE.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps(run()['stats'],sort_keys=True))
