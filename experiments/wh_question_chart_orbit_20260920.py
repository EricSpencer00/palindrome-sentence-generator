"""Typed wh-question complements with explicit extraction sites."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs"/"wh-question-chart-orbit-20260920.json"; EXPERIMENT_ID="wh-question-chart-orbit-20260920"

def letters(text:str)->str: return re.sub(r"[^a-z]", "", text.casefold())
def audit(text:str)->dict[str,object]:
    t=letters(text); bad=[(i,len(t)-i-1) for i in range(len(t)//2) if t[i]!=t[-i-1]]; f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and not bad,"first_mismatch":bad[0] if bad else None,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def consume(left:str,right:str)->tuple[str,str]|None:
    n=min(len(left),len(right))
    if n and left[:n]!=right[-n:][::-1]: return None
    return left[n:],right[:-n] if n else right

@dataclass(frozen=True)
class Item:
    symbol:str; text:str; valency:str; number:str|None=None; agreement:str|None=None; extraction:str|None=None
def it(symbol,text,valency,**kw): return Item(symbol,text,valency,**kw)

def expand(symbol:str,*,depth=0)->list[tuple[Item,...]]:
    if depth>6:return []
    if symbol=="SPEAKER": return [(it("SPEAKER","the bard","speaker",number="singular"),),(it("SPEAKER","the guards","speaker",number="plural"),),(it("SPEAKER","a raven","speaker",number="singular"),)]
    if symbol=="WH": return [(it("WH","which letter","wh-object",extraction="object"),),(it("WH","what sign","wh-object",extraction="object"),),(it("WH","which vow","wh-object",extraction="object"),)]
    if symbol=="SUBJ": return [(it("SUBJ","the king","question-subject",number="singular"),),(it("SUBJ","the queens","question-subject",number="plural"),),(it("SUBJ","a friend","question-subject",number="singular"),)]
    if symbol=="WHQ":
        rows=[]
        for subj in expand("SUBJ",depth=depth+1):
            plural=subj[0].number=="plural"; forms=(("do","read","plural"),("do","guard","plural")) if plural else (("does","read","singular"),("does","keep","singular"))
            for aux,verb,agr in forms:
                for wh in expand("WH",depth=depth+1):
                    rows.append(wh+(it("AUX",aux,"question-aux",agreement=agr),)+subj+(it("QVERB",verb,"extraction-predicate",agreement=agr),))
        return rows
    if symbol=="DIALOGUE":
        rows=[]
        for sp in expand("SPEAKER",depth=depth+1):
            for pred in ("asks","wonders"):
                for q in expand("WHQ",depth=depth+1): rows.append(sp+(it("SPEECH",pred,"speech-valency"),)+q)
        return rows
    if symbol=="SCENE":
        rows=list(expand("DIALOGUE",depth=depth+1))
        for l in expand("DIALOGUE",depth=depth+1):
            for bridge in ("and","while"):
                for r in expand("DIALOGUE",depth=depth+1): rows.append(l+(it("BRIDGE",bridge,"coordination"),)+r)
        return rows
    return []

def complete_derivations():
    seen=set(); rows=[]
    for p in expand("SCENE"):
        k=tuple(x.text for x in p)
        if k not in seen:seen.add(k);rows.append(p)
    return rows

def run(*,state_limit=300_000):
    paths=complete_derivations(); states=pruned=advances=0; candidates=[]; witnesses=[]
    def pair(lp,rp):
        nonlocal states,pruned,advances
        def walk(li,ri,left,right,ls,rs,env):
            nonlocal states,pruned,advances
            if states>=state_limit:return
            if li>=len(lp) and ri<0:
                if left or right:return
                ordered=ls+tuple(reversed(rs));rendered=" ".join(x.text for x in ordered);checked=audit(rendered)
                if checked["exact"]: candidates.append({"rendered":rendered,"audit":checked,"provenance":{"construction":"typed wh-question extraction chart","symbols":[x.symbol for x in ordered],"valencies":[x.valency for x in ordered],"extraction_sites":[x.extraction for x in ordered],"variable_word_boundaries":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False},"reader_status":"unreviewed; exactness does not certify readability"})
                return
            states+=1
            if li>=len(lp) or ri<0:return
            l,r=lp[li],rp[ri];e=dict(env)
            if l.valency=="question-subject":e["ln"]=l.number or ""
            if r.valency=="question-subject":e["rn"]=r.number or ""
            if l.valency=="question-aux" and l.agreement!=e.get("ln"):pruned+=1;return
            if r.valency=="question-aux" and r.agreement!=e.get("rn"):pruned+=1;return
            if l.extraction and l.valency!="wh-object":pruned+=1;return
            residual=consume(left+letters(l.text),letters(r.text)+right)
            if residual is None:
                pruned+=1
                if len(witnesses)<20:
                    z=" ".join(x.text for x in ls+(l,)+(r,)+tuple(reversed(rs)));witnesses.append({"rendered":z,"depth":li,"audit":audit(z),"reader_status":"diagnostic chart witness"})
                return
            advances+=1;walk(li+1,ri-1,residual[0],residual[1],ls+(l,),(r,)+rs,e)
        walk(0,len(rp)-1,"","",(),(),{})
    controls=[{"rendered":" ".join(x.text for x in p),"audit":audit(" ".join(x.text for x in p)),"reader_status":"complete generated wh-question control; not exact"} for p in paths[:8]]
    for lp in paths:
        for rp in paths:
            pair(lp,rp)
            if states>=state_limit:break
        if states>=state_limit:break
    candidates.sort(key=lambda x:x["audit"]["letters"],reverse=True)
    result={"experiment":EXPERIMENT_ID,"method":"typed wh-question extraction sites in character chart","complete_prose_controls":controls,"candidates":candidates,"witnesses":witnesses,"stats":{"grammar_paths":len(paths),"states":states,"pruned":pruned,"chart_advances":advances,"exact":len(candidates)},"provenance":{"complete_wh_questions":True,"extraction_sites_typed":True,"subject_auxiliary_agreement":True,"variable_word_boundaries":True,"independent_pointer_sha_audit":True,"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"aligned_token_mirror":False,"novelty_preflight":"new wh-extraction complement family","next_construction":"add pied-piping prepositional wh complements with typed extraction sites"}}
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+"\n");return result
if __name__=="__main__":print(json.dumps(run(),indent=2))
