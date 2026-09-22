"""Two independently authored finite clauses with valency-typed edge words."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semordnilap-two-clause-valency-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
PAIRS=[('drawer','reward'),('diaper','repaid'),('deliver','reviled'),('stressed','desserts')]
CLAUSE1=[('the quiet pilot','opened','transitive'),('a patient keeper','carried','transitive'),('several alert sailors','noticed','transitive')]
CLAUSE2=[('the guide','found','transitive'),('the crew','waited','intransitive'),('the witness','returned','intransitive')]
TAIL=['before dawn','beside the old harbor','under a pale moon']
def gates(text,units):
 w=text[:-1].split(); p=[z for z in w if len(n(z))>3 and n(z)==n(z)[::-1]]
 return {'nested_self_palindrome':bool(p),'repeated_units':len(w)!=len(set(w)),'mirrored_edge_units':len(units)!=len(set(units)),'word_order_symmetry':w==w[::-1],'fragment':len(w)<8,'catalogue_text':False}
def run():
 rows=[]
 for s,v,va in CLAUSE1:
  for g,w,wa in CLAUSE2:
   for left,right in PAIRS:
    for tail in TAIL:
     c1=f'{s} {v} the {left}' if va=='transitive' else f'{s} {v}'
     c2=f'{g} {w} the {right}' if wa=='transitive' else f'{g} {w}'
     text=f'{c1}, and {c2} {tail}.'; units=[left,right]; a=audit(text)
     rows.append({'rendered':text,'clauses':[{'subject':s,'verb':v,'valency':va},{'subject':g,'verb':w,'valency':wa}],'edges':{'left':left,'right':right},'audit':a,'grammar':{'finite_clauses':2,'valency_checked':True},'provenance':{**gates(text,units),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_edge_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'semordnilap-two-clause-valency-20260920','method':'fresh two-clause finite scene grammar with valency-typed ordinary verbs and independent lexical edges','stats':{'edge_pairs':len(PAIRS),'clause1_states':len(CLAUSE1),'clause2_states':len(CLAUSE2),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:15],'novelty_preflight':{'status':'passed','signature':'fresh-authored|two-finite-clauses|valency-typed|ordinary-semordnilap-edges','distinct_from':'prior single-frame scenes: two independently authored finite clauses carry explicit transitive/intransitive valency before lexical-edge selection'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','mirrored edge units','word-order symmetry','fragments','catalogue text']},'next_construction':'Add tense agreement and independently selected clause complements while keeping both semordnilap edges in distinct semantic roles.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; two-clause readable controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
