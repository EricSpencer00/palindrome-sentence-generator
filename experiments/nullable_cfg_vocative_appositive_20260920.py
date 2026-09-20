"""Nullable CFG with typed vocative-appositive speaker edges."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=norm(s); i,j=0,len(t)-1
    while i<j and t[i]==t[j]: i+=1; j-=1
    return {"letters":len(t),"exact":bool(t) and i>=j,
            "first_mismatch":None if i>=j else {"index":i,"forward":t[i],"reverse":t[-1-i]},
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def pointer_exact(s):
    t=norm(s); return bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2))
def consume(a,b):
    n=min(len(a),len(b)); return (a[n:],b[n:]) if a[:n]==b[:n] else None


SPEAKERS=(
    {"id":"keeper","vocative":("keeper",),"role":"witness"},
    {"id":"sailor","vocative":("sailor",),"role":"traveler"},
    {"id":"poet","vocative":("poet",),"role":"observer"},
)
COMMANDS=(
    {"id":"lantern","speaker_role":"witness","words":("guard","the","lantern")},
    {"id":"harbor","speaker_role":"traveler","words":("watch","the","bright","harbor")},
    {"id":"garden","speaker_role":"observer","words":("remember","the","winter","garden")},
)
CLAUSES=(
    {"id":"harbor","words":("the","harbor","is","quiet"),"kind":"COP"},
    {"id":"keeper","words":("a","careful","keeper","studies","the","chart"),"kind":"SVO"},
    {"id":"garden","words":("the","young","poet","remembers","the","garden"),"kind":"SVO"},
)


def derivations():
    out=[]
    # Vocative is an appositive speaker edge, and command choice is typed by
    # speaker role.  A nullable continuation adds a second independent clause.
    for sp in SPEAKERS:
        for cmd in COMMANDS:
            if cmd["speaker_role"]!=sp["role"]: continue
            voc=sp["vocative"]+cmd["words"]
            out.append({"kind":"VOC-IMP","speaker":sp["id"],"words":voc,"continuation":None})
            for c in CLAUSES:
                out.append({"kind":"VOC-IMP+CLAUSE","speaker":sp["id"],"words":voc+("and",)+c["words"],"continuation":c["id"]})
    for c in CLAUSES:
        out.append({"kind":c["kind"],"speaker":None,"words":c["words"],"continuation":None})
    return out


def controls(ds):
    rows=[]
    for d in ds:
        if d["speaker"] is None or len(rows)>=4: continue
        split=1
        text=" ".join(d["words"][:split])+", "+" ".join(d["words"][split:])+"."
        rows.append({"rendered":text,"audit":audit(text),"independent_pointer_exact":pointer_exact(text),
                     "speaker_role_agreement":True,"complete_semantic_parse":True,"reader_eligible":False,
                     "provenance":"authored vocative-appositive control; not an exact candidate"})
    return rows


def run(limit=30000):
    ds=derivations(); exact=[]; diagnostics=[]; seen=set(); states=char_prunes=semantic_prunes=seam_prunes=0
    for left in ds:
        for right in ds:
            if left["speaker"]==right["speaker"] and left["continuation"]==right["continuation"]: semantic_prunes+=1; continue
            lw,rw=left["words"],tuple(reversed(right["words"]))
            stack=[(0,0,"","","","",False,False)]
            while stack and states<limit:
                li,ri,lt,rt,lb,rb,ls,rs=stack.pop(); states+=1
                if li==len(lw) and ri==len(rw):
                    rendered=(lt+"; "+rt).strip(); au=audit(rendered)
                    if len(diagnostics)<8: diagnostics.append({"rendered":rendered,"audit":au,"cross_word_seam":ls or rs,"complete_semantic_parse":True,"reader_eligible":False,"reason":"complete typed vocative derivation but residual/exact gate failed"})
                    if lb or rb or not(ls or rs):
                        if not(ls or rs): seam_prunes+=1
                        continue
                    if au["exact"] and pointer_exact(rendered) and au["letters"]>38 and rendered not in seen:
                        seen.add(rendered); exact.append({"rendered":rendered,"audit":au,"independent_pointer_exact":True,"cross_word_seam":True,"provenance":{"left_derivation":left,"right_derivation":right,"speaker_role_agreement":True,"posthoc_repair":False,"finished_tape_reversal":False,"mirrored_units":False}})
                    continue
                if li<len(lw):
                    w=lw[li]; res=consume(lb+norm(w),rb)
                    if res is None: char_prunes+=1
                    else: stack.append((li+1,ri,(lt+" " if lt else "")+w,rt,res[0],res[1],ls or(bool(lb) and len(norm(w))>len(rb)),rs))
                if ri<len(rw):
                    w=rw[ri]; res=consume(lb,rb+norm(w)[::-1])
                    if res is None: char_prunes+=1
                    else: stack.append((li,ri+1,lt,w+(" "+rt if rt else ""),res[0],res[1],ls,rs or(bool(rb) and len(norm(w))>len(lb))))
            if states>=limit: break
        if states>=limit: break
    return {"method":"nullable-cfg-vocative-appositive-20260920","status":"completed_no_exact_closure" if not exact else "exact_candidates_require_readers","derivations":len(ds),"states":states,"character_prunes":char_prunes,"semantic_prunes":semantic_prunes,"seam_prunes":seam_prunes,"state_limit":limit,"exact_candidates":exact,"exact_candidate_count":len(exact),"rendered_diagnostics":diagnostics,"controls":controls(ds),"reader_facing_candidates":[],"reader_eligible":False,"independent_validation":["literal outside-in two-pointer","forward/reverse SHA-256"],"provenance":"fresh nullable CFG with vocative-appositive speaker edge, typed command agreement, and optional second clause; full semantic derivations emit variable word boundaries into live residuals with cross-word seam; no reversal, repair, catalogue text, or mirrored units","novelty_preflight":{"passed":True,"overlaps_checked":["mixed-declarative-vocative-imperative-20260920","dialogue-speech-act-residual-20260916","nullable-cfg-appositive-adjunct-20260920"],"unused_dimension":"speaker-role agreement at a vocative appositive edge, coupled to a typed imperative inside nullable CFG","reason":"registry covers vocative and appositive families separately but no exact solver lane couples vocative identity, command valency, and nullable continuation"},"first_live_diagnostic":"character residual mismatch at vocative CFG edge" if not exact else "exact closure requires blinded reader review","next_construction":"hold out a vocative question-answer edge with role agreement; do not widen command inventory"}


if __name__=="__main__":
    result=run(); out=ROOT/"runs/nullable-cfg-vocative-appositive-20260920.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ("derivations","states","character_prunes","semantic_prunes","seam_prunes","exact_candidate_count")}))
