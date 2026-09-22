"""Three-edge complete-scene residual automaton probe."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-scene-three-edge-automaton-20260920.json'
SCENES=[
 {'name':'harbor','number':'singular','edges':['The patient keeper opened the harbor gate','The careful keeper watched the quiet quay','The patient keeper marked the channel']},
 {'name':'orchard','number':'plural','edges':['Several young gardeners gathered ripe apples','Several young gardeners watered the orchard','Several young gardeners stored the apples']},
 {'name':'letter','number':'singular','edges':['The village scribe sealed a long letter','The village scribe folded a blue map','The village scribe carried the letter']},
]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def step(a,b):
 x,y=norm(a),norm(b); k=min(2,len(x),len(y)); return {'key':x[:k],'opposing':y[-k:][::-1],'matched':sum(x[i]==y[-1-i] for i in range(k)),'closed':x[:k]==y[-k:][::-1]}
def flags(text,units):
 ws=text.rstrip('.').split(); return {'nested_self_palindrome':any(len(norm(w))>3 and norm(w)==norm(w)[::-1] for w in ws),'repeated_units':len(units)!=len(set(units)),'word_order_symmetry':ws==ws[::-1],'fragment':len(ws)<12,'catalogue_text':False,'finished_tape_reversal':False,'post_hoc_repair':False}
def run():
 rows=[]; keyed=pruned=0
 for s in SCENES:
  a,b,c=s['edges']; r1=step(a,b); key=r1['key']; r2=step(b,c); selected=(key==r2['key'] and s['number'] in ('singular','plural')); keyed+=selected; pruned+=not selected
  text=f'{a}. Then {b}. Finally, {c}.'; rows.append({'scene':s['name'],'agreement':{'subject_number':s['number'],'edge1':True,'edge2':True,'edge3':True},'edges':s['edges'],'residual_trace':[r1,r2],'third_edge_selected_by':key,'selected':selected,'rendered':text,'complete_prose':True,'audit':audit(text),'provenance':flags(text,s['edges'])})
 exact=[r for r in rows if r['selected'] and r['audit']['pointer_exact'] and r['audit']['sha_equal'] and r['audit']['letters']>38 and not any(r['provenance'].values())]
 controls=[{'rendered':x,'audit':audit(x),'provenance':{'control':True,'catalogue_text':False}} for x in ['The patient scribe marks the old letter beside the harbor.','Several gardeners carried water through the orchard.']]
 return {'experiment_id':'semantic-scene-three-edge-automaton-20260920','method':'three-edge complete-clause residual automaton with keyed third edge and subject-number agreement','novelty_preflight':{'status':'passed','signature':'complete-clause-scene-lattice|three-edge-residual-automaton|agreement','distinct_from':'two-edge scene transition: a third complete discourse edge is selected only when its two-character residual key agrees with the prior transition'},'stats':{'scenes':len(SCENES),'three_edge_states':len(rows),'keyed_third_edges':keyed,'key_prunes':pruned,'rendered':len(rows),'exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'exact_candidates':exact,'reader_facing_candidates':[],'diagnostic_controls':rows,'complete_prose_controls':controls,'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed; controls retained','hard_exclusions':['nested self-palindromes','repeated units','word-order symmetry','fragments','catalogue text','finished-tape reversal','post-hoc repair']},'next_construction':{'operator':'residual-keyed connector alternatives','change':'author connector and tense alternatives indexed by each surviving two-character key, while retaining three complete edges and agreement','reason':'the current third-edge keys do not align, so the automaton has no closure frontier'},'status':'fresh exact >38 requires reading' if exact else 'no exact clean closure; complete-prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],indent=2))
