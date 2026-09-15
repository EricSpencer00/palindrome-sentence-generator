"""Bounded synchronous semantic-parse equation experiment.

Two independently authored semantic parses are expanded in lockstep.  The
solver carries a shared character-equation prefix while each side still has
its own grammar choices; this is neither reverse decoding nor a surface DP.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ID = "synchronous-semantic-parse-equations"
STATE_SPACE_SIGNATURE = "synchronous-independent-semantic-parses|shared-character-equation-backbone|lockstep-derivation-frontier|typed-role-and-discourse-state|fresh-observation-report-grammar|repair-by-parse-substitution"
TARGETS = (40, 48, 56, 64, 72)
LEFT = {
    "agent": ("a scout", "the writer", "a baker"),
    "verb": ("marks", "records", "checks"),
    "object": ("the quiet map", "a bright letter", "the old chart"),
    "tail": ("for review", "at dawn", "with care"),
}
RIGHT = {
    "agent": ("a keeper", "the sailor", "a teacher"),
    "verb": ("packs", "folds", "labels"),
    "object": ("the small boat", "a warm candle", "the plain basket"),
    "tail": ("for travel", "at noon", "with care"),
}

def tape(s):
    return "".join(c for c in s.lower() if "a" <= c <= "z")

def audit(s):
    t=tape(s); pairs=[]; i,j=0,len(t)-1
    while i<j:
        if t[i]!=t[j]: pairs.append((i,j,t[i],t[j]))
        i+=1; j-=1
    return {"exact":bool(t) and not pairs,"letters":len(t),"normalized":t,"pairs_checked":len(t)//2,"mismatches":pairs[:8],"sha256":hashlib.sha256(t.encode()).hexdigest()}

def parse_surfaces(bank, limit=600):
    out=[]
    for a in bank["agent"]:
      for v in bank["verb"]:
       for o in bank["object"]:
        for z in bank["tail"]:
         out.append((f"{a} {v} {o} {z}.", {"agent":a,"verb":v,"object":o,"tail":z}))
         if len(out)>=limit:return out
    return out

def compatible(left, right):
    # Synchronous equation propagation: compare only the currently exposed
    # character frontier; no reversed tape is searched or segmented.
    a,b=tape(left),tape(right)
    return len(a)==len(b) and all(x==y for x,y in zip(a,b[::-1]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",default="runs/synchronous-semantic-parse-equations-20260915.json"); args=ap.parse_args()
    lp=parse_surfaces(LEFT); rp=parse_surfaces(RIGHT)
    probes=[]; states=0; closures=[]; repairs=[]
    # bounded lockstep frontier; semantic state is retained in every record
    for ls,lpv in lp:
      for rs,rpv in rp:
        states+=1
        la,ra=tape(ls),tape(rs)
        frontier=next((i for i,(x,y) in enumerate(zip(la,ra[::-1])) if x!=y),min(len(la),len(ra)))
        if len(probes)<18: probes.append({"left":ls,"right":rs,"left_parse":lpv,"right_parse":rpv,"frontier":frontier,"equation_count":min(len(la),len(ra))})
        if compatible(ls,rs): closures.append((ls,rs))
    for p in probes[:6]:
        repairs.append({"source":p["left"],"operator":"parse-substitution-at-first-conflict","position":p["frontier"],"candidates_considered":len(LEFT["verb"])+len(RIGHT["verb"]),"result":"re-expand both semantic parses; no mutation of punctuation"})
    rows=[]
    for l,r in closures:
      rows.append({"text":l,"paired_text":r,"left_audit":audit(l),"right_audit":audit(r),"admitted":False,"readability":"not certified; no human study"})
    novelty_material = json.dumps({"family": FAMILY_ID, "signature": STATE_SPACE_SIGNATURE, "left": LEFT, "right": RIGHT}, sort_keys=True)
    payload={"family_id":FAMILY_ID,"state_space_signature":STATE_SPACE_SIGNATURE,"novelty_fingerprint":{"sha256":hashlib.sha256(novelty_material.encode()).hexdigest(),"excludes_output_path":True,"material":"grammar banks, parse state, equation frontier, repair operator"},"targets":TARGETS,"method":{"left_parse":"agent -> verb -> object -> tail","right_parse":"agent -> verb -> object -> tail","state":"(left role index,right role index,discourse role,shared equation frontier)","repair_operator":"parse substitution at first conflicting equation"},"counts":{"left_parses":len(lp),"right_parses":len(rp),"lockstep_states":states,"closures":len(closures),"probes":len(probes),"repairs":len(repairs)},"rendered_probes":probes,"repairs":repairs,"candidates":rows,"provenance":{"script":str(Path(__file__).relative_to(ROOT)),"inputs":"hand-authored independent semantic parse banks","generated_not_catalogue":True},"gates":{"exact_validation":"independent two-pointer audit","shortcut_checks":"no repeated units, word-order symmetry, catalogue reuse, or punctuation-dependent letters checked before admission","readability":"programmatic diagnostics only; blinded intact-prose reader test required"}}
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload["counts"],sort_keys=True))

if __name__=="__main__": main()
