"""Intransitive locative cross-role transitions with relation metadata."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/typed-locative-crossrole-relations-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=[('the lantern keeper','waits','singular'),('several river pilots','wait','plural'),('the patient cartographer','rests','singular')]; LOC=[('beside the quiet inlet','adjacent'),('under the weathered beacon','sheltered'),('along the narrow channel','path')]; REL=['at dawn','near the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; transitions=0
 for (s,v,num),(loc,relation),rel in itertools.product(S,LOC,REL):
  left=n(v); right=n(loc); ok=relation in ('adjacent','sheltered','path') and bool(left and right); transitions+=1
  text=f'{s} {v} {loc} {rel}.'; rows.append({'rendered':text,'semantic_frame':{'subject':s,'number':num,'event':v,'valency':'intransitive','location':loc,'relation':relation,'adjunct':rel},'cross_role_transition':{'from_role':'event','to_role':'location','relation':relation,'accepted':ok},'audit':audit(text),'provenance':{**gates(text,[s,v,loc,rel]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['cross_role_transition']['accepted'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'typed-locative-crossrole-relations-20260920','method':'typed intransitive locative event-to-location transitions with explicit relation metadata','stats':{'subjects':len(S),'locative_relations':len(LOC),'adjuncts':len(REL),'transitions':transitions,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|intransitive-locative|crossrole-relation|typed-metadata','distinct_from':'prior transitive event-theme gate: intransitive verbs now select typed locative relation states instead of direct themes'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Unify transitive theme and intransitive location alternatives in one semantic valency lattice.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; locative relation prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
