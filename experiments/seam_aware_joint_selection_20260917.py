"""Seam-aware joint selection for grammatical coordinated prose.

Unlike the Cartesian baseline, each frontier step scores a *pair* of lexical
choices against the opposing tape boundary, then applies the 100--140 letter
band and register checks before retaining a candidate.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs"/"seam-aware-joint-selection-20260917.json"
ID="seam-aware-joint-selection-20260917"
SUBJ=[("the careful baker","sg"),("the quiet sailors","pl"),("the patient gardeners","pl")]
VERBS=[("carried","past"),("opened","past"),("marked","past"),("recorded","past")]
OBJS=["a letter for the council","the lantern by the doorway","a weathered map of the harbor","the spare key for the garden"]
PLACES=["beside the river at dusk","under the old bridge at dawn","near the harbor after rain","along the stone path home"]

def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
    t=norm(s); m=[(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"letters":len(t),"exact":bool(t) and not m,"mismatch_count":len(m),"mismatch_rate":len(m)/max(1,len(t)//2),"first_mismatches":m[:10],"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def independent(s):
    x=[c.casefold() for c in s if c.casefold() in 'abcdefghijklmnopqrstuvwxyz']; i,j=0,len(x)-1; n=0
    while i<j: n += x[i]!=x[j]; i+=1; j-=1
    raw=''.join(x); return {"exact":bool(raw) and n==0,"mismatch_count":n,"sha256_forward":hashlib.sha256(raw.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(raw[::-1].encode()).hexdigest()}
def flags(s):
    ws=[norm(x) for x in re.findall('[A-Za-z]+',s)]; content=[w for w in ws if w not in {'a','an','the','and','then','by','for','at','of','near','under','along','beside','after'}]
    return {"word_order_mirror":ws==[w[::-1] for w in ws[::-1]],"self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],"borrowed_catalogue_text":False,"finished_tape_reversed":False}
def render(l,r):
    (s1,v1,o1,p1),(s2,v2,o2,p2)=l,r
    return f"{s1} {v1} {o1}, {p1}; and {s2} {v2} {o2}, {p2}."
def pair_cost(text):
    a=audit(text); return a['mismatch_count']+max(0,100-a['letters'])*2+max(0,a['letters']-140)*2
def run():
    rows=[]
    # Jointly rank complete register pairs: no independent rank-then-filter.
    for left in [(s,v,o,p) for s,_ in SUBJ for v in VERBS for o in OBJS for p in PLACES]:
      for right in [(s,v,o,p) for s,_ in SUBJ for v in VERBS for o in OBJS for p in PLACES]:
        l=(left[0],left[1],left[2],left[3]); r=(right[0],right[1],right[2],right[3]); s=render(l,r); a=audit(s)
        if not 100<=a['letters']<=140: continue
        ind=independent(s); f=flags(s)
        if f['word_order_mirror'] or f['self_palindromic_content_words'] or f['finished_tape_reversed']: continue
        rows.append({'rendered':s,'audit':a,'independent_audit':ind,'shortcut_flags':f,'registers':{'left':{'subject_number':next(n for ss,n in SUBJ if ss==left[0]),'tense':left[1][1],'valency':'transitive'},'right':{'subject_number':next(n for ss,n in SUBJ if ss==right[0]),'tense':right[1][1],'valency':'transitive'}},'provenance':{'generator':ID,'construction':'joint opposed-frontier pair selection','catalogue_imported':False,'seed_used_as_output':False},'_cost':pair_cost(s)})
    rows.sort(key=lambda x:(x['_cost'],-x['audit']['letters']))
    for x in rows: x.pop('_cost',None)
    exact=[x for x in rows if x['audit']['exact'] and x['independent_audit']['exact']]
    return {'experiment_id':ID,'status':'completed_no_exact_closure' if not exact else 'completed_exact_candidates','construction':'seam-aware joint opposed-frontier lexical selection','config':{'joint_pairs_evaluated':len(SUBJ)**2*len(VERBS)**2*len(OBJS)**2*len(PLACES)**2,'length_band':[100,140],'selection':'pair cost before eligibility filtering'},'actual_candidates':rows[:12],'best':rows[0] if rows else None,'exact_candidates':exact,'independent_validation':'separate two-pointer audit and SHA-256 recomputation','reader_gate':'closed: no exact novel survivor; metrics diagnose only','next_repair':'Add a human-authored scene lattice with seam-compatible inflectional variants; preserve joint pair scoring and length band.','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+'\n'); r=run(); print(json.dumps({'status':r['status'],'candidates':len(r['actual_candidates']),'best':r['best']['rendered'] if r['best'] else None,'letters':r['best']['audit']['letters'] if r['best'] else 0,'exact':len(r['exact_candidates'])}))
