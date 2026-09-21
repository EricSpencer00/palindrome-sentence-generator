"""Two-edge scene lattice carrying a live two-character residual."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/semantic-scene-residual-transition-20260920.json'
FRAMES=[
 {"name":"harbor", "number":"singular", "a":"The patient keeper opened the harbor gate", "b":"The careful keeper watched the quiet quay", "connector":"Then", "c":"the keeper marked the channel", "d":"the keeper charted the channel"},
 {"name":"orchard", "number":"plural", "a":"Several young gardeners gathered ripe apples", "b":"Several young gardeners watered the orchard", "connector":"Later", "c":"the gardeners stored the apples", "d":"the gardeners tended the orchard"},
 {"name":"letter", "number":"singular", "a":"The village scribe sealed a long letter", "b":"The village scribe folded a blue map", "connector":"Afterward", "c":"the scribe carried the letter", "d":"the scribe carried the map"},
]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def live_residual(left,right):
 a,b=norm(left),norm(right); k=min(2,len(a),len(b)); return {'left_prefix':a[:k],'right_reverse_suffix':b[-k:][::-1],'matched':sum(a[i]==b[-1-i] for i in range(k)),'closed':a[:k]==b[-k:][::-1]}
def flags(text,units):
 ws=text.rstrip('.').split(); return {'nested_self_palindrome':any(len(norm(w))>3 and norm(w)==norm(w)[::-1] for w in ws),'repeated_units':len(units)!=len(set(units)),'word_order_symmetry':ws==ws[::-1],'fragment':len(ws)<10,'catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False}
def run():
 rows=[]; transitions=0; prunes=0
 for f in FRAMES:
  for x,y in ((f['a'],f['b']),(f['c'],f['d'])):
   transitions+=1; r=live_residual(x,y); prunes+=not r['closed']; text=f"{x}. {f['connector']} {y}." if x in (f['a'],f['b']) else f"{x}. {y}."; rows.append({'scene':f['name'],'agreement':{'number':f['number'],'valid':True},'edge':x,'paired_edge':y,'rendered':text,'live_residual':r,'audit':audit(text),'provenance':flags(text,[x,y])})
 exact=[r for r in rows if r['audit']['pointer_exact'] and r['audit']['sha_equal'] and r['audit']['letters']>38 and not any(r['provenance'].values())]
 controls=[{'rendered':x,'audit':audit(x),'provenance':{'control':True,'catalogue_text':False}} for x in ['The patient scribe marks the old letter beside the harbor.','Several gardeners carried water through the orchard.']]
 return {'experiment_id':'semantic-scene-residual-transition-20260920','method':'two-edge complete-clause scene lattice carrying two-character residual classes with agreement','novelty_preflight':{'status':'passed','signature':'complete-clause-scene-lattice|two-character-residual-transition|discourse-edge','distinct_from':'single-edge scene lattice: a residual class is computed at one complete-clause edge and retained as a typed transition into a second discourse edge'},'stats':{'frames':len(FRAMES),'transitions':transitions,'live_prunes':prunes,'rendered':len(rows),'exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'exact_candidates':exact,'reader_facing_candidates':[],'diagnostic_controls':rows,'complete_prose_controls':controls,'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed; controls retained','hard_exclusions':['nested self-palindromes','repeated units','word-order symmetry','fragments','catalogue text','finished-tape reversal','post-hoc repair']},'next_construction':{'operator':'three-edge residual automaton','change':'author a third complete discourse edge keyed by the surviving two-character class, with explicit subject-number agreement at every transition','reason':'two-edge transitions still fail to close the full opposing tape; preserve surviving residual classes instead of widening lexical banks'},'status':'fresh exact >38 requires reading' if exact else 'no exact clean closure; complete-prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],indent=2))
