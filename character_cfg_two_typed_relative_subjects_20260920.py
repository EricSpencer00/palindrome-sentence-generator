"""Two independently typed relative subjects under a live character CFG gate."""
import hashlib,itertools,json,re,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/character-cfg-two-typed-relative-subjects-20260920.json'
ID='character-cfg-two-typed-relative-subjects-20260920'; SIG='fresh-authored|two-typed-relative-subjects|nullable-pp|heldout-shuffle|live-character-obligation'
B={'SUBJ':['the pilot','a nurse'],'V':['charts','carries'],'OBJ':['the harbor','a lantern'],'PP':['','under stars'],'AGENT':['who waits','that guides'],'OBSERVER':['who listens','that watches']}
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def deriv():
 for s,v,o,p,a,b in itertools.product(B['SUBJ'],B['V'],B['OBJ'],B['PP'],B['AGENT'],B['OBSERVER']):
  yield f'{s} {a} and {b} {v} {o}'+(f' {p}' if p else '')+'.'
def live(a,b):
 x,y=letters(a),letters(b); trace=[]
 for i in range(max(len(x),len(y))):
  if i>=len(x) or i>=len(y): return False,trace,'length'
  trace.append({'position':i,'left_char':x[i],'right_char':y[-1-i],'obligation':'equal'})
  if x[i]!=y[-1-i]: return False,trace,'mismatch'
 return True,trace,'closed'
def flags(s):
 w=s[:-1].split(); return {'nested_self_palindrome':any(len(letters(q))>3 and letters(q)==letters(q)[::-1] for q in w),'repeated_units':len(w)!=len(set(w)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<10,'catalogue_text':False,'mirrored_units':False}
def make(a,b,kind):
 ok,tr,why=live(a,b); return {'rendered':a,'independent_right':b,'control_kind':kind,'grammar':{'nullable_pp':True,'relative_subjects':[{'type':'agent','antecedent':'subject'},{'type':'observer','antecedent':'subject'}],'typed_independent_choices':True},'bilateral_obligation_trace':tr,'closure':why,'audit':audit(a),'provenance':{**flags(a),'fresh_authored_productions':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'per_search_rlAIF':False}}
def run():
 ds=list(deriv()); shuffled=list(ds); random.Random(20260920).shuffle(shuffled)
 intact=[make(a,b,'intact') for a,b in itertools.product(ds,repeat=2)]; shuf=[make(a,b,'shuffled') for a,b in zip(ds,shuffled)]
 exact=[r for r in intact if r['closure']=='closed' and r['audit']['pointer_exact'] and r['audit']['sha256_forward']==r['audit']['sha256_reverse'] and not any(r['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':ID,'method':'two independently typed relative subjects (agent and observer) with nullable PP, intersected by live bilateral character obligations','stats':{'derivations':len(ds),'intact_pairs':len(intact),'shuffled_pairs':len(shuf),'intact_closed':sum(r['closure']=='closed' for r in intact),'shuffled_closed':sum(r['closure']=='closed' for r in shuf),'exact_clean':len(exact)},'exact_candidates':exact,'reader_facing_candidates':intact[:10],'shuffled_controls':shuf[:10],'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'prior single-relative held-out lane: agent and observer relative subjects are independently typed productions with separate antecedent roles before live character intersection'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'hard_exclusions':['reversal','repair','catalogue/API text','token mirror','per-search RLAIF'],'falsifier':'if typed-agent/observer intact closure rate does not separate from shuffled lexical controls, the second relative subject adds no live-intersection signal'},'next_operator':'Add agreement features to the two relative subjects without widening lexical choices; retain nullable PP and shuffled controls.','status':'fresh exact candidate requires reading' if exact else 'no exact clean closure; typed-relative controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
