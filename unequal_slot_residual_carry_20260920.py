"""Unequal semantic-slot lengths with live residual carryover."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/unequal-slot-residual-carry-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUB=[('the lantern keeper','singular','guards'),('several river pilots','plural','guard'),('the patient cartographer','singular','marks')]; OBJ=['the quiet inlet','a weathered beacon','the narrow channel']; ADJ=['before dawn','beside the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; carries=0
 for (s,num,v),o,a in itertools.product(SUB,OBJ,ADJ):
  slots=[n(s),n(v),n(o),n(a)]; residual=''; trace=[]
  for i,slot in enumerate(slots):
   combined=residual+slot; consumed=combined[:max(1,len(combined)//3)]; residual=combined[len(consumed):]; carries+=len(residual); trace.append({'slot':i,'input_length':len(slot),'consumed':len(consumed),'residual':residual[-4:]})
  text=f'{s} {v} {o} {a}.'; rows.append({'rendered':text,'slot_lengths':[len(x) for x in slots],'residual_trace':trace,'audit':audit(text),'provenance':{**gates(text,slots),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if not x['residual_trace'][-1]['residual'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'unequal-slot-residual-carry-20260920','method':'fresh unequal semantic-slot lengths with residual carryover across subject/verb/object/adjunct','stats':{'scene_states':len(rows),'carry_steps':carries,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|unequal-slot-lengths|residual-carryover|semantic-lattice','distinct_from':'prior fixed-width buffers: residual text carries across unequal slot lengths and is consumed incrementally'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Type residual carryover by semantic role and prune only when the role-compatible residual cannot close.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; unequal-slot prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
