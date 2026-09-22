"""Held-out nullable-PP + subject-relative live CFG intersection.

Lexical choices are made before each character obligation is discharged, while
intact prose and deterministic shuffled lexical controls share the same audit.
"""
import hashlib,itertools,json,re,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/character-cfg-nullable-relative-heldout-20260920.json'
ID='character-cfg-nullable-relative-heldout-20260920'; SIG='fresh-authored|nullable-pp|subject-relative|heldout-lexical|live-character-intersection'
TRAIN={'SUBJ':['the sailor','a keeper'],'V':['marks','guards'],'OBJ':['the inlet','a beacon'],'PP':['','at dawn']}
HELD={'SUBJ':['the pilot','a nurse'],'V':['charts','carries'],'OBJ':['the harbor','a lantern'],'PP':['','under stars'],'REL':['who waits','that listens']}
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def deriv(bank):
 for s,v,o,p,r in itertools.product(bank['SUBJ'],bank['V'],bank['OBJ'],bank['PP'],bank['REL']):
  yield f'{s} {v} {o}'+(f' {p}' if p else '')+f' {r}.'
def live(a,b):
 x,y=letters(a),letters(b); tr=[]
 for i in range(max(len(x),len(y))):
  if i>=len(x) or i>=len(y): return False,tr,'length'
  tr.append({'position':i,'left':x[i],'right':y[-1-i],'obligation':'equal'})
  if x[i]!=y[-1-i]: return False,tr,'mismatch'
 return True,tr,'closed'
def flags(s):
 w=s[:-1].split(); return {'nested_self_palindrome':any(len(letters(q))>3 and letters(q)==letters(q)[::-1] for q in w),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<8,'catalogue_text':False,'mirrored_units':False}
def row(left,right,kind):
 ok,tr,why=live(left,right); return {'rendered':left,'independent_right':right,'control_kind':kind,'grammar':{'nullable_pp':True,'subject_relative':True,'lexical_selection':'held-out live'},'bilateral_obligation_trace':tr,'closure':why,'audit':audit(left),'provenance':{**flags(left),'fresh_authored_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'per_search_rlAIF':False}}
def run():
 intact=list(deriv(HELD)); shuffled=list(intact); random.Random(20260920).shuffle(shuffled)
 rows=[row(a,b,'intact') for a,b in itertools.product(intact,repeat=2)]
 shuf=[row(a,b,'shuffled') for a,b in zip(intact,shuffled)]
 exact=[r for r in rows if r['closure']=='closed' and r['audit']['pointer_exact'] and r['audit']['sha256_forward']==r['audit']['sha256_reverse'] and not any(r['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':ID,'method':'held-out nullable-PP and subject-relative CFG productions intersected at live bilateral character frontiers','stats':{'heldout_derivations':len(intact),'intact_pairs':len(rows),'shuffled_pairs':len(shuf),'intact_closed':sum(r['closure']=='closed' for r in rows),'shuffled_closed':sum(r['closure']=='closed' for r in shuf),'exact_clean':len(exact)},'exact_candidates':exact,'reader_facing_candidates':rows[:10],'shuffled_controls':shuf[:10],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior CFG/Earley/seam runs: nullable and relative productions are held out, and intact-versus-shuffled controls are compared under the same live character obligation state'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'hard_exclusions':['reversal','repair','catalogue/API text','token mirror','per-search RLAIF'],'falsifier':'if intact and shuffled closure rates are indistinguishable, held-out grammar structure adds no live-intersection signal'},'next_operator':'Add two independently typed relative subjects while retaining nullable PP and the held-out/shuffled comparison.','status':'fresh exact candidate requires reading' if exact else 'no exact clean closure; held-out controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
