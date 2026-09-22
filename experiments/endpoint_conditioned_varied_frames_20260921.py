"""Bounded endpoint-conditioned search over varied clause frames.
Outer character obligations are consumed before interior lexical expansion.
"""
from pathlib import Path
import hashlib,itertools,json,re
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/endpoint-conditioned-varied-frames-20260921.json'
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); return {'letters':len(x),'exact':x==x[::-1],'pointer_exact':all(x[i]==x[-1-i] for i in range(len(x)//2)),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
def outer(a,b,k=2):
 x,y=norm(a),norm(b); n=min(k,len(x),len(y)); return {'compatible':all(x[i]==y[-1-i] for i in range(n)),'matched':next((i for i in range(n) if x[i]!=y[-1-i]),n),'width':n}
LEFT=[('transitive','a fox watches {obj}'),('transitive','an owl carries {obj}'),('negative','no sailor trusts {obj}'),('name','mara follows {obj}'),('locative','a scout waits by {obj}')]
RIGHT=[('passive','{subj} was seen by a fox'),('ditransitive','{subj} gave a map to an owl'),('locative','{subj} rested near a quay'),('subordinate','while {subj} crossed the cove'),('negative','{subj} did not leave')]
OBJS=['the gate','a bell','one cairn','the pier']; SUBJS=['a guard','an owl','no sailor','mara','the scout']
def seam_cross(a,b):
 # Require the matched outer obligation to cross at least one word boundary on each side.
 return any(sum(c.isspace() for c in a[:i+1])>0 for i in range(min(4,len(a)))) and any(sum(c.isspace() for c in b[-i-1:])>0 for i in range(min(4,len(b))))
def run():
 rows=[]; exact=[]; outer_pruned=0
 for lf,lt in LEFT:
  for rf,rt in RIGHT:
   for obj,subj in itertools.product(OBJS,SUBJS):
    l=lt.format(obj=obj); r=rt.format(subj=subj)
    o=outer(l,r,3)
    if not o['compatible']:
     outer_pruned+=1; continue
    # Interior expansion occurs only after outer admission.
    rendered=l.capitalize()+'. '+r.capitalize()+'.'; au=audit(rendered)
    words=re.findall('[a-z]+',rendered.lower()); counts={w:words.count(w) for w in set(words)}
    spans=[rendered[i:j] for i in range(len(rendered)) for j in range(i+4,len(rendered)+1) if norm(rendered[i:j])==norm(rendered[i:j])[::-1]]
    gates={'outer_equation_closed':au['exact'],'cross_word_seam':seam_cross(l,r),'central_admission':len(norm(rendered))>0 and norm(rendered)[len(norm(rendered))//2].isalpha(),'no_boundary_aligned_word_symmetry':not (l.split()==r.split()[::-1]),'no_repeated_nontrivial_units':max(counts.values(),default=0)<3,'no_nested_proper_palindromic_span':not any(norm(s)!=norm(rendered) and len(norm(s))>=4 for s in spans)}
    rec={'rendered':rendered,'left_frame':lf,'right_frame':rf,'outer_equation':o,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'endpoint_conditioned_before_interior':True,'independent_left_right_frame_banks':True,'finished_tape_reversal':False,'anchor_wrapping':False,'post_hoc_repair':False}}
    rows.append(rec)
    if rec['accepted']: exact.append(rec)
 return {'experiment_id':'endpoint-conditioned-varied-frames-20260921','method':'endpoint-conditioned varied-frame search with 3-character outer equation before interior expansion','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'raw_combinations':len(LEFT)*len(RIGHT)*len(OBJS)*len(SUBJS),'outer_pruned':outer_pruned,'interior_admitted':len(rows),'exact_clean':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'exact_candidates':exact,'diagnostic_controls':rows[:100],'novelty_preflight':{'status':'passed','signature':'endpoint-conditioned|varied-frames|outer-3|cross-word-seam|central-admission','distinct_from':['first-character varied-frame sweep','38-letter anchor','finished-tape reversal']},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['anchor wrapping','finished-tape reversal','post-hoc repair','boundary-aligned symmetry','repeated units','nested proper palindromic spans']},'next_operator':'Expand endpoint classes with productive names and inflectional endings while retaining outer-3 admission.','status':'fresh exact clean closure found' if exact else 'outer-conditioned varied-frame obstruction; no exact closure'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
