"""Fresh live-buffer search over optional adjunct/PP/relative sentence frames."""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/live-buffer-optional-grammar-search-20260920.json"; EXPERIMENT_ID="live-buffer-optional-grammar-search-20260920"
def letters(text): return re.sub(r"[^a-z]", "", text.casefold())
def audit(surface):
    tape=letters(surface); bad=next(((i,len(tape)-1-i) for i in range(len(tape)//2) if tape[i]!=tape[-1-i]),None)
    return {"letters":len(tape),"exact":bool(tape) and bad is None,"first_mismatch":bad,"sha256_forward":hashlib.sha256(tape.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest()}
@dataclass(frozen=True)
class Slot: role:str; words:tuple[str,...]
def slot(role,*words): return Slot(role,tuple(dict.fromkeys(w.casefold() for w in words)))
def compatible(left,right):
    n=min(len(left),len(right)); return left[:n]==right[::-1][:n]
def search(template,limit=128,state_limit=900000):
    found=[]; rejected=0; states=0
    def distinct(w,ls,rs): return w not in ls and w not in rs and w!=w[::-1]
    def finish(words):
        checked=audit(" ".join(words))
        if checked["exact"] and len(set(words))==len(words): found.append({"rendered":" ".join(words),"audit":checked,"provenance":{"construction":"complete optional-slot live-buffer grammar","template_roles":[s.role for s in template],"finished_tape_reversal":False,"post_hoc_repair":False,"catalogue_text":False,"repeated_word":False},"reader_status":"unreviewed"})
    def walk(lo,hi,pref,suff,ls,rs):
        nonlocal states,rejected
        if states>=state_limit or len(found)>=limit:return
        if lo>hi: finish(ls+rs); return
        if lo==hi:
            for w in template[lo].words:
                if distinct(w,ls,rs) and compatible(pref+letters(w),suff): states+=1; finish(ls+(w,)+rs)
                else: rejected+=1
            return
        for lw in template[lo].words:
            if not distinct(lw,ls,rs):continue
            for rw in template[hi].words:
                if not distinct(rw,ls,rs) or rw==lw:continue
                states+=1; lp,rp=pref+letters(lw),letters(rw)+suff
                if compatible(lp,rp):walk(lo+1,hi-1,lp,rp,ls+(lw,),(rw,)+rs)
                else:rejected+=1
    walk(0,len(template)-1,"","",(),()); found.sort(key=lambda x:x["audit"]["letters"],reverse=True)
    return {"candidates":found[:limit],"stats":{"states":states,"rejected":rejected,"exact":len(found)}}
def build_templates():
    d=slot("determiner","a","an","the","some","one"); agent=slot("agent","aide","artist","baker","clerk","farmer","keeper","nurse","poet","pilot","scribe","teacher","writer"); verb=slot("verb","aids","bakes","calls","draws","edits","feeds","gives","helps","keeps","marks","mends","names","opens","reads","rips","saves","sends","teaches","writes"); noun=slot("object","book","chart","letter","memo","map","note","plan","poem","record","sign","story","tale","text","verse"); plural=slot("subject","artists","bakers","clerks","farmers","keepers","nurses","poets","pilots","scribes","teachers","writers"); pverb=slot("plural_verb","admire","bake","call","draw","edit","feed","give","help","keep","mark","mend","name","open","read","save","send","teach","write"); prep=slot("preposition","at","by","for","in","near","on","with"); place=slot("place","camp","desk","garden","harbor","lane","market","park","room","school","shore"); rel=slot("relative","who","that"); names=slot("name","ada","anna","ava","diana","iris","lena","maria","maya","nina","nora","sara","zoe"); adjunct=slot("adjunct","today","often","quietly","there","inside","outside")
    return {"base":(d,agent,verb,d,noun,d,plural,pverb,names),"pp":(d,agent,verb,d,noun,prep,d,place,d,plural,pverb,names),"relative":(d,agent,rel,verb,d,noun,d,plural,pverb,names),"pp_relative_adjunct":(d,agent,rel,verb,d,noun,prep,d,place,d,plural,pverb,names,adjunct)}
CONTROLS=("the baker reads a book","a poet writes a poem","the nurse helps a farmer","artists draw a map","the clerk marks a note","a pilot saves a plan","the teacher gives a lesson","writers edit a story","the keeper opens a gate","a scribe copies a letter","the artist paints a sign","farmers mend a fence","the poet names a child","a nurse calls the doctor","clerks read a record","the pilot flies near shore","teachers help the writer","a baker feeds a child","the farmer visits a market","poets share a verse")
def run():
    searches={n:search(t) for n,t in build_templates().items()}; candidates=[{**c,"template":n} for n,r in searches.items() for c in r["candidates"]]
    return {"experiment_id":EXPERIMENT_ID,"method":"character-by-character full unmatched buffers over complete optional PP/relative/adjunct frames","searches":searches,"candidates":candidates,"exact_candidates":[c for c in candidates if c["audit"]["exact"]],"best_length":max((c["audit"]["letters"] for c in candidates),default=0),"controls":[{"surface":s,"audit":audit(s),"natural":True} for s in CONTROLS],"control_count":len(CONTROLS),"reader_gate":"closed until complete exact candidates are independently reviewed","next_construction":"Represent optional slots as recursive clause adjunction with lexicalized multiword constituents if no readable >38 candidate emerges."}
if __name__=="__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"best_length":result["best_length"],"exact":len(result["exact_candidates"]),"controls":result["control_count"]}))
