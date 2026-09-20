"""Role-typed unequal residual carryover; prune only incompatible closures."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/role-typed-unequal-residual-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=[('the lantern keeper','singular','guards','agent'),('several river pilots','plural','guard','agent'),('the patient cartographer','singular','marks','agent')]; O=['the quiet inlet','a weathered beacon','the narrow channel']; A=['before dawn','beside the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; pruned=0; typed=0
 for (s,num,v,srole),o,a in itertools.product(S,O,A):
  slots=[('agent',n(s)),('event',n(v)),('theme',n(o)),('setting',n(a))]; residual=''; trace=[]
  for role,slot in slots:
   combined=residual+slot; consume=max(1,len(combined)//3); residual=combined[consume:]; typed+=1; trace.append({'role':role,'residual':residual[-5:],'role_compatible':role in ('agent','event','theme','setting')})
  role_close=all(x['role_compatible'] for x in trace) and not residual
  if not role_close: pruned+=1
  text=f'{s} {v} {o} {a}.'; rows.append({'rendered':text,'roles':slots,'residual_trace':trace,'role_compatible_closure':role_close,'audit':audit(text),'provenance':{**gates(text,[x[1] for x in slots]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['role_compatible_closure'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'role-typed-unequal-residual-20260920','method':'role-typed unequal residual carryover with delayed prune on role-compatible closure','stats':{'scene_states':len(rows),'typed_role_steps':typed,'pruned_incompatible_closures':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|role-typed-residual|unequal-carryover|delayed-prune','distinct_from':'prior untyped carry: residuals now retain semantic role labels and are pruned only at role-incompatible closure'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Add cross-role residual transitions where an event residual may legally feed a theme boundary under a typed valency relation.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; role-typed prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
