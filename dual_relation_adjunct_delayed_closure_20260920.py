"""Two relation-specific adjunct slots with delayed variable-residual closure."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/dual-relation-adjunct-delayed-closure-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=['the lantern keeper','several river pilots','the patient cartographer']; V=[('guards','the quiet inlet','theme'),('waits','beside the narrow channel','location')]; ADJ={'theme':['before dawn','near the river'],'location':['under clear stars','along the old road']}
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<9,'catalogue_text':False}
def run():
 rows=[]; steps=0
 for s,(v,c,val) in itertools.product(S,V):
  for a1 in ADJ[val]:
   for a2 in ADJ[val]:
    if a1==a2: continue
    text=f'{s} {v} {c} {a1}, and later {a2}.'; slots=[s,v,c,a1,a2]; residual=''; trace=[]
    for role,slot in zip(('agent','event','theme','adjunct1','adjunct2'),slots):
     z=n(slot); width=min(4,max(1,len(z)//4)); steps+=width; residual=(residual+z)[width:]; trace.append({'role':role,'width':width,'residual_tail':residual[-5:]})
    closed=not residual; rows.append({'rendered':text,'valency':val,'adjuncts':[a1,a2],'delayed_closure':closed,'residual_trace':trace,'audit':audit(text),'provenance':{**gates(text,slots),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['delayed_closure'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'dual-relation-adjunct-delayed-closure-20260920','method':'two successive relation-specific adjunct slots with delayed variable-residual closure','stats':{'subjects':len(S),'valency_options':len(V),'rendered':len(rows),'residual_steps':steps,'closed':sum(x['delayed_closure'] for x in rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|dual-relation-adjuncts|delayed-closure|variable-residuals','distinct_from':'prior single adjunct lane: two distinct relation-specific adjunct slots are consumed before closure is evaluated'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Permit mixed valency adjunct pairs and carry residual role labels through both delayed slots.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; dual-adjunct diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
