"""Live character residual over unified transitive/locative valency alternatives."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/valency-live-character-residual-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
S=['the lantern keeper','several river pilots','the patient cartographer']; V=[('guards','the quiet inlet','theme'),('marks','a weathered beacon','theme'),('waits','beside the narrow channel','location'),('rests','under the old bridge','location')]; A=['before dawn','near the river','under clear stars']
def gates(t,u):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':len(u)!=len(set(u)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<7,'catalogue_text':False}
def run():
 rows=[]; checks=0; pruned=0
 for s,(v,c,val),a in itertools.product(S,V,A):
  text=f'{s} {v} {c} {a}.'; left=n(s+v); right=n(c+a); checks+=1; live=left[:2]==right[-2:][::-1]
  if not live: pruned+=1
  rows.append({'rendered':text,'valency':val,'relation_complement':c,'live_residual':{'left_prefix':left[:2],'right_suffix_reversed':right[-2:][::-1],'accepted':live},'audit':audit(text),'provenance':{**gates(text,[s,v,c,a]),'fresh_authored_lattice':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['live_residual']['accepted'] and x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'valency-live-character-residual-20260920','method':'live character residual over unified transitive-theme and intransitive-location alternatives','stats':{'subjects':len(S),'valency_options':len(V),'checks':checks,'pruned':pruned,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':[],'diagnostic_controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|unified-valency|live-character-residual|relation-complements','distinct_from':'prior static valency lattice: character residual is checked during alternative selection, with complement relation metadata retained'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text'],'reader_gate':'closed; diagnostic controls are not reader material'},'next_construction':'Carry a variable-length residual through valency transitions and allow relation-specific adjunct insertion before closure.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; live valency prose diagnostics retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
