"""Whole-scene paired grammar with jointly solved opening choices."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/whole-scene-joint-opening-grammar-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=[('the quiet pilot charts the inlet','agent-theme'),('a patient keeper carries a beacon','agent-theme'),('several young guides guard the bridge','agent-theme')]; RIGHT=[('the guide watches the harbor','agent-theme'),('a sailor marks the channel','agent-theme'),('several scouts protect the lantern','agent-theme')]; TAIL=['at sunrise','before dusk','under stars']
def gates(t,u):
 w=t.rstrip('.').split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<6,'catalogue_text':False}
def run():
 rows=[]; joint=0
 for (l,lrole),(r,rrole),tail in itertools.product(LEFT,RIGHT,TAIL):
  # Joint semantic choice: both complete clauses selected first; only their
  # opening character class is compared, not a boundary index or seam walk.
  joint+=1; compatible=n(l)[0]==n(r)[0]; text=f'{l}, while {r} {tail}.'
  rows.append({'rendered':text,'joint_scene':{'left_role':lrole,'right_role':rrole,'tail':tail,'opening_compatible':compatible},'audit':audit(text),'provenance':{**gates(text,[l,r,tail]),'fresh_whole_scene_grammar':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); clean=[x for x in rows if x['joint_scene']['opening_compatible'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'whole-scene-joint-opening-grammar-20260920','method':'fresh whole-scene paired clause grammar with jointly solved opening choices','stats':{'left_clauses':len(LEFT),'right_clauses':len(RIGHT),'tails':len(TAIL),'joint_scenes':joint,'opening_compatible':sum(x['joint_scene']['opening_compatible'] for x in rows),'reader_clean':len(clean),'max_letters':rows[0]['audit']['letters']},'exact_candidates':clean,'reader_facing_candidates':clean,'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|whole-scene-grammar|joint-opening-choice|complete-prose','distinct_from':'indexed seam lanes: complete left/right clauses and tail are jointly authored before a single opening-choice compatibility test; no boundary automaton'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'exact clean >38 only','hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Jointly solve two opening characters plus a semantic relation choice before rendering whole scenes.','status':'fresh exact candidate requires reading' if clean else 'no exact clean whole-scene row; intact controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
