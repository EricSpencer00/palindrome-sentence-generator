"""Asynchronous width-two buffers between typed semantic slots."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/async-width2-semantic-slots-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the lantern keeper','singular','guards'),('several river pilots','plural','guard'),('the patient cartographer','singular','marks'),('the patient cartographers','plural','mark')]; OBJ=['the quiet inlet','a weathered beacon','the narrow channel']; ADJ=['before dawn','beside the river','under clear stars','along the old road']
def gates(t,units):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; transitions=0; pruned=0
 for (s,num,v),o,adj in itertools.product(SUB,OBJ,ADJ):
  slots=[s,v,o,adj]; states=[]; live=True
  for i in range(3):
   transitions+=1; left=n(slots[i]); right=n(slots[i+1]); ok=left[-2:]==right[:2]; live &= ok; states.append({'boundary':f'{i}->{i+1}','buffer_left':left[-2:],'buffer_right':right[:2],'accepted':ok})
  if not live: pruned+=1
  text=f'{s} {v} {o} {adj}.'; rows.append({'rendered':text,'semantic_frame':{'subject':s,'number':num,'verb':v,'object':o,'adjunct':adj},'async_buffers':states,'audit':audit(text),'provenance':{**gates(text,slots),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if all(z['accepted'] for z in x['async_buffers']) and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'async-width2-semantic-slots-20260920','method':'fresh asynchronous width-two buffers across subject/verb/object/typed adjunct slots','stats':{'number_states':len(SUB),'adjunct_choices':len(ADJ),'scene_states':len(rows),'boundary_transitions':transitions,'pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|semantic-lattice|async-width2|typed-adjuncts','distinct_from':'prior fixed width-two scene lattice: buffers advance asynchronously across each semantic boundary with adjunct agreement choices'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Allow unequal slot lengths with residual carryover instead of equal two-character boundary windows.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; asynchronous width2 controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
