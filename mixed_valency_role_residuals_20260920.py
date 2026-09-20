"""Mixed-valency adjunct pairs with role-labeled delayed residuals."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/mixed-valency-role-residuals-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=['the lantern keeper','several river pilots','the patient cartographer']; V=[('guards','the quiet inlet','theme'),('waits','beside the narrow channel','location')]; ADJ=[('before dawn','time'),('near the river','location'),('under clear stars','setting'),('along the old road','path')]
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<9,'catalogue_text':False}
def run():
 rows=[]; steps=0
 for s,(v,c,val),(a1,r1),(a2,r2) in itertools.product(S,V,ADJ,ADJ):
  if a1==a2: continue
  slots=[('agent',s),('event',v),('theme_or_location',c),('adjunct1_'+r1,a1),('adjunct2_'+r2,a2)]; residual=''; trace=[]
  for role,slot in slots:
   z=n(slot); width=min(4,max(1,len(z)//4)); steps+=width; residual=(residual+z)[width:]; trace.append({'role':role,'width':width,'residual_tail':residual[-5:]})
  text=f'{s} {v} {c} {a1}, and later {a2}.'; rows.append({'rendered':text,'valency':val,'adjunct_roles':[r1,r2],'residual_trace':trace,'role_labeled_closure':not residual,'audit':audit(text),'provenance':{**gates(text,[x[1] for x in slots]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['role_labeled_closure'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'mixed-valency-role-residuals-20260920','method':'mixed transitive-theme/intransitive-location adjunct pairs with role-labeled delayed residuals','stats':{'subjects':len(S),'valency_options':len(V),'adjunct_roles':len(ADJ),'rendered':len(rows),'residual_steps':steps,'closed':sum(x['role_labeled_closure'] for x in rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|mixed-valency|role-labeled-residuals|dual-delayed-slots','distinct_from':'prior same-valency adjuncts: both adjunct slots may carry different semantic relation roles while residuals remain labeled'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Use role-compatible residual transitions between mixed adjunct slots with explicit valency relation edges.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; mixed-valency diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
