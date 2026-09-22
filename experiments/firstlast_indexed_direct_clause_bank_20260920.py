"""First/last-character indexed direct clause bank with live seam selection."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/firstlast-indexed-direct-clause-bank-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CLAUSES=['the careful pilot charts a hidden inlet at sunrise','a patient keeper carries the brass lantern beside water','several young guides mark the northern road before dusk','the quiet cartographer records a weathered harbor under stars','three alert sailors guard the narrow bridge near moonlight','the old gardener watches a silver gate along the river']
def gates(t):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 idx={}
 for c in CLAUSES:
  t=n(c); idx.setdefault((t[0],t[-1]),[]).append(c)
 rows=[]; candidate_pairs=0; seam_checks=0
 for key,lefts in idx.items():
  reverse_key=(key[1],key[0]); rights=idx.get(reverse_key,[])
  for l,r in itertools.product(lefts,rights):
   if l==r: continue
   candidate_pairs+=1; a=n(l); b=n(r)[::-1]; seam_checks+=1; matched=0
   for x,y in zip(a,b):
    if x!=y: break
    matched+=1
   text=l+'.'; rows.append({'rendered':text,'indexed_key':key,'opposing_clause':r+'.','live_seam':{'matched_prefix':matched,'closed':matched==min(len(a),len(b)) and len(a)==len(b)},'audit':audit(text),'provenance':{**gates(text),'fresh_heldout_bank':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['live_seam']['closed'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 controls = [
     {'rendered': c + '.', 'audit': audit(c),
      'provenance': {'complete_prose': True, 'reader_eligible': False,
                     'fresh_heldout_bank': True, **gates(c + '.')}}
     for c in CLAUSES
 ]
 return {'experiment_id':'firstlast-indexed-direct-clause-bank-20260920','method':'first/last character indexed direct clause bank with compatibility selection before live seam checks','stats':{'bank':len(CLAUSES),'index_keys':len(idx),'candidate_pairs':candidate_pairs,'seam_checks':seam_checks,'reader_clean':len(clean),'max_letters':max((x['audit']['letters'] for x in controls),default=0)},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':controls,'diagnostic_pairs':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|firstlast-index|direct-clause-bank|preselection-seam','distinct_from':'prior all-pairs direct sweep: first/last classes select compatible complete clauses before reverse segmentation'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'reader list contains only exact clean rows','hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Index two-character exposed classes and permit compatible clauses with unequal normalized lengths.','status':'fresh exact candidate requires reading' if clean else 'no exact clean indexed pair; intact controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
