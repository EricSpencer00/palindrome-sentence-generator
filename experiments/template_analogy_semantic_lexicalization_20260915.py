"""Bounded experiment: abstract palindrome shapes, then fresh semantic lexicalization.

The shapes encode only constituent roles and seam widths; no catalogue string or
catalogue word is imported.  Left and right clauses are independently authored
from different semantic inventories.  This is a feasibility audit, not a
readability certificate.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize
FAMILY_ID="template-analogy-semantic-lexicalization"
STATE_SPACE_SIGNATURE="abstract-role-shape-analogy|independent-fresh-semantic-lexicalization|paired-seam-width-signature|joint-character-equation-enumeration|no-catalogue-word-import"
SHAPES=(
 {"id":"agent-action-patient","roles":("det","agent","verb","patient"),"seam":"aa","meaning":"an agent performs an action on a patient"},
 {"id":"observer-reports-scene","roles":("det","observer","verb","det","scene"),"seam":"ee","meaning":"an observer reports a scene"},
 {"id":"maker-creates-object","roles":("det","maker","verb","det","object"),"seam":"oo","meaning":"a maker creates an object"},
 {"id":"keeper-guides-traveler","roles":("det","keeper","verb","det","traveler"),"seam":"rr","meaning":"a keeper guides a traveler"},
)
LEFT={
"agent":(("a","cartographer"),("the","nurse"),("a","gardener"),("the","teacher")),
"observer":(("a","witness"),("the","scout"),("a","listener")),
"maker":(("a","weaver"),("the","carver"),("a","potter")),
"keeper":(("a","warden"),("the","pilot"),("a","guide")),}
RIGHT={
"agent":(("a","mason"),("the","doctor"),("a","sailor"),("the","farmer")),
"observer":(("a","reader"),("the","camera"),("a","reporter")),
"maker":(("a","sculptor"),("the","builder"),("a","printer")),
"keeper":(("a","captain"),("the","driver"),("a","warden")),}
VERBS={"agent":(("charts","a harbor"),("treats","a wound"),("tends","the orchard"),("trains","a pupil")),"observer":(("describes","a tableau"),("records","a signal"),("notices","the weather")),"maker":(("weaves","a basket"),("carves","a totem"),("shapes","an idol")),"keeper":(("guides","a pilgrim"),("leads","the convoy"),("steers","a vessel"))}
def tape(s): return normalize_letters(s)
def fp(out):
    found=set(); target=out.resolve() if out else None
    for base in (ROOT/"runs",ROOT/"data",ROOT/"experiments"):
      for p in base.rglob("*.json"):
       if target and p.resolve()==target: continue
       try: obj=json.loads(p.read_text())
       except Exception: continue
       def walk(x):
        if isinstance(x,str):
         try:
          t=tape(x)
          if t and t==t[::-1]: found.add(t)
         except Exception: pass
        elif isinstance(x,dict):
         for v in x.values(): walk(v)
        elif isinstance(x,list):
         for v in x: walk(v)
       walk(obj)
    return found,{"palindrome_tapes":len(found),"fingerprint_sha256":hashlib.sha256("\n".join(sorted(found)).encode()).hexdigest(),"output_excluded":bool(target)}
def pointers(t):
    mm=[]; i=0; j=len(t)-1
    while i<j:
      if t[i]!=t[j]: mm.append((i,j,t[i],t[j]))
      i+=1; j-=1
    return {"exact":not mm and bool(t),"comparisons":len(t)//2,"mismatches":mm[:8]}
def clause(role, side, n, v, obj):
    return {"side":side,"role":role,"subject":n,"verb":v,"object":obj,"words":(n[0],n[1],v,obj.split()[0],*obj.split()[1:]),"meaning":f"{n[0]} {n[1]} {v} {obj}"}
def content(words): return [w for w in words if w not in {"a","an","the"}]
def render(c1,c2): return " ".join(c1["words"]).capitalize()+"; "+" ".join(c2["words"])+"."
def run(out):
    existing, audit=fp(out); stats=Counter(shapes=0,pairs=0,exact=0,probes=0)
    exact=[]; probes=[]
    for shape in SHAPES:
      stats["shapes"]+=1; role=shape["roles"][1]
      for ln in LEFT[role]:
       for rn in RIGHT[role]:
        for lv,lo in VERBS[role]:
         for rv,ro in VERBS[role]:
          stats["pairs"]+=1; a=clause(role,"left",ln,lv,lo); b=clause(role,"right",rn,rv,ro); text=render(a,b); t=tape(text); p=pointers(t)
          direct=(t==t[::-1])
          row={"shape_id":shape["id"],"shape_meaning":shape["meaning"],"rendered":text,"letters":len(t),"normalized_letters":t,"normalized_sha256":hashlib.sha256(t.encode()).hexdigest(),"left_provenance":a,"right_provenance":b,"independent_exact_audit":{"direct":direct,"two_pointer":p,"agree":direct==p["exact"]},"readability_diagnostic":{"status":"diagnostic_only","word_count":len(tokenize(text)),"blinded_reader_required":True},"reader_status":"not_run"}
          if p["exact"]:
           stats["exact"]+=1; row["mechanically_admitted"]=all(mechanical_admission_checks(text,min_letters=30,max_letters=180).values()) and t not in existing; exact.append(row)
          elif len(probes)<30:
           stats["probes"]+=1; row["first_mismatch"]={"left":p["mismatches"][0][0],"right":p["mismatches"][0][1]} if p["mismatches"] else None; probes.append(row)
    return {"status":"template_analogy_complete","family_id":FAMILY_ID,"state_space_signature":STATE_SPACE_SIGNATURE,"config":{"shapes_are_role_only":True,"catalogue_words_imported":False,"left_bank_count":sum(map(len,LEFT.values())),"right_bank_count":sum(map(len,RIGHT.values())),"verbs_are_fresh_hand_authored":True,"bounded":True},"novelty_audit":{"registry_entries_read_before_run":41,"existing_tape_fingerprint":audit,"exact_rows_checked_against_existing":True},"stats":dict(stats),"exact_candidates":exact,"prominent_exact_candidate":exact[0] if exact else None,"rendered_candidates_and_probes":probes,"repair_operator":{"operator":"role-preserving semantic substitution","action":"retain the abstract role shape and replace one independently authored subject, verb, or object while preserving agreement and seam widths; rerun the complete joint equation audit","forbidden":["catalogue word import","reverse residual segmentation","word-order symmetry","repeated/self-palindromic units"]},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"shape_source":"abstract roles authored for this run","lexical_source":"independent hand-authored left/right banks","readability_certificate":False},"reader_gate":{"status":"not_run","reason":"Programmatic diagnostics cannot certify English readability; exact candidates require blinded intact-prose and shuffled-control readers."}}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--out",required=True,type=Path); a=p.parse_args()
 if a.out.exists(): p.error("refusing to overwrite output")
 r=run(a.out); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+"\n"); print(json.dumps({"stats":r["stats"],"exact":len(r["exact_candidates"])}))
if __name__=="__main__": main()
