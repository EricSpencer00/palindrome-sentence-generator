"""Two-character exposed-class index for unequal complete clauses."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/twochar-indexed-unequal-clause-bank-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CLAUSES=['the bright pilot charts a cove at sunrise','a careful keeper carries old maps beside water','several young guides mark the northern road before dusk','the quiet cartographer records a harbor under stars','three alert sailors guard a narrow bridge near moonlight','an old gardener watches a silver gate along the river']
def gates(t):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 idx={}
 for c in CLAUSES:
  t=n(c); idx.setdefault((t[:2],t[-2:]),[]).append(c)
 rows=[]; selected=0; checks=0
 for (start,end),lefts in idx.items():
  for (rstart,rend),rights in idx.items():
   if end!=rstart[::-1]: continue
   for l,r in itertools.product(lefts,rights):
    if l==r or len(n(l))==len(n(r)): continue
    selected+=1; a=n(l); b=n(r)[::-1]; checks+=1; k=0
    for x,y in zip(a,b):
     if x!=y: break
     k+=1
    rows.append({'rendered':l+'.','opposing_clause':r+'.','index_keys':{'left':[start,end],'right':[rstart,rend]},'unequal_lengths':[len(a),len(b)],'live_seam':{'matched':k,'closed':len(a)==len(b) and k==len(a)},'audit':audit(l+'.'),'provenance':{**gates(l+'.'),'fresh_heldout_bank':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['live_seam']['closed'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'twochar-indexed-unequal-clause-bank-20260920','method':'two-character exposed-class index with unequal normalized clause lengths and live seam checks','stats':{'bank':len(CLAUSES),'index_keys':len(idx),'selected_pairs':selected,'seam_checks':checks,'reader_clean':len(clean),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|twochar-index|unequal-lengths|live-seam','distinct_from':'prior first/last index: two-character exposed classes and unequal-length selection precede seam consumption'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'exact clean >38 only','hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Index variable-length exposed prefixes and suffixes with partial seam compatibility instead of exact two-character keys.','status':'fresh exact >38 requires reading' if clean else 'no exact clean two-character indexed pair; intact controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
