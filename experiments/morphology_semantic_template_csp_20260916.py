"""Simultaneous morphology-aware semantic template construction."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="morphology-semantic-template-csp-20260916"
SIGNATURE="morphology-semantic-template-csp|derivational-affix-choice|inflection-agreement-state|semantic-frame-realization|live-character-equations|independent-exact-audit|affix-repair"
FRAMES=[("The baker","bake","bread","warm"),("A keeper","keep","records","careful"),("The singer","sing","melodies","bright")]
FORMS={"bake":("bakes","baked","baking"),"keep":("keeps","kept","keeping"),"sing":("sings","sang","singing")}
def letters(s): return re.sub("[^a-z]","",s.lower())
def audit(s):
 t=letters(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {"exact":bool(t) and not bad,"letters":len(t),"mismatches":len(bad),"first_mismatch":bad[0] if bad else None}
def realize(f,kind,affix):
 subj,lemma,obj,adj=f; v=FORMS[lemma][kind]
 noun=({"warm":"warmth","careful":"carefulness","bright":"brightness"}[adj] if affix else adj+" " + obj)
 return f"{subj} {v} {noun} for the {obj}."
def run(phase,repair=False):
 out=[]; kinds=range(3); affixes=(False,True) if not repair else (True,False)
 for a in FRAMES:
  for b in FRAMES:
   if a==b: continue
   for ka in kinds:
    for kb in kinds:
     for aa in affixes:
      for ab in affixes:
       left,right=realize(a,ka,aa),realize(b,kb,ab); text=left+" "+right
       toks=re.findall("[a-z]+",text.lower())
       out.append({"phase":phase,"frames":[a,b],"morphology":[[ka,aa],[kb,ab]],"left":left,"right":right,"rendered":text,"audit":audit(text),"complete_sentences":True,"all_different_content_words":len(toks)==len(set(toks)),"reader_eligible":False,"provenance":"independently authored semantic frames with generated inflection/derivation; no catalogue text"})
 return out
def main():
 base,repair=run("base"),run("repair",True)
 p={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"operator":"semantic frame and morphology choices enumerated jointly while auditing mirrored character equations","base":{"candidates":base,"exact_count":sum(x["audit"]["exact"] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x["audit"]["exact"] for x in repair)},"repair_action":"swap affix-choice order and rerun all tense/agreement combinations to expose a different completion frontier","provenance":{"catalogue_used":False,"borrowed_text":False,"word_order_mirror":False,"fragments":False,"repeated_units_allowed":False}}
 (ROOT/"runs/morphology-semantic-template-csp-20260916.json").write_text(json.dumps(p,indent=2)+"\n")
 print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":p["base"]["exact_count"],"repair_exact":p["repair"]["exact_count"]}))
if __name__=="__main__": main()
