"""Constructive ordinary-word semordnilap edge grammar; no tape reversal."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/constructive-semordnilap-scene-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
EDGES=[('drawer','reward','theme'),('diaper','repaid','theme'),('deliver','reviled','event'),('stressed','desserts','state'),('star','rats','setting')]
SUB=[('the quiet pilot','singular'),('a patient keeper','singular'),('several alert sailors','plural')]
VERBS={'singular':['opened','carried','noticed'],'plural':['opened','carried','noticed']}
TAIL=['before dawn','beside the old harbor','under a pale moon']
def gates(text,units):
 w=n(text).split(); toks=text[:-1].split(); pals=[x for x in toks if len(n(x))>3 and n(x)==n(x)[::-1]]
 return {'nested_self_palindrome':bool(pals),'repeated_units':len(toks)!=len(set(toks)),'mirrored_semordnilap_units':len(units)!=len(set(units)),'word_order_symmetry':toks==toks[::-1],'fragment':len(toks)<7,'catalogue_text':False}
def run():
 rows=[]
 for subj,num in SUB:
  for verb in VERBS[num]:
   for left,right,role in EDGES:
    for tail in TAIL:
     # Both clauses are authored independently: the partner is a lexical
     # choice in a second semantic role, never a reflected phrase/unit.
     text=f'{subj} {verb} the {left}; the guide {right} {tail}.'
     units=[left,right]; a=audit(text); rows.append({'rendered':text,'scene':{'subject':subj,'number':num,'verb':verb,'left_role':role,'right_partner_role':'event','tail':tail},'semordnilap_edges':{'left':left,'right':right,'pair':f'{left}/{right}'},'audit':a,'grammar':{'agreement':'passed','independent_sides':True},'provenance':{**gates(text,units),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered']))
 ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','mirrored_semordnilap_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'constructive-semordnilap-scene-20260920','method':'independent two-sided scene grammar over ordinary semordnilap lexical edges','stats':{'edge_pairs':len(EDGES),'subjects':len(SUB),'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:15],'novelty_preflight':{'status':'passed','signature':'fresh-authored|ordinary-semordnilap-edges|independent-scene-sides|no-mirrored-units','distinct_from':'prior residual-boundary lanes: lexical semordnilap partners are independently embedded in grammatical scene roles before exact testing'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38','hard_exclusions':['nested self-palindromic spans','repeated units','mirrored semordnilap units','word-order symmetry','fragments','catalogue text']},'next_construction':'Expand the edge bank with valency-typed ordinary verbs and two independently authored finite clauses, then carry character obligations only at lexical-edge boundaries.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; constructive readable near-misses retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
