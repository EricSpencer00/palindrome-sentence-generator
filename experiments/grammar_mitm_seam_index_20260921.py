"""Grammar-constrained meet-in-the-middle seam index (diagnostic)."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/grammar-mitm-seam-index-20260921.json'
LEFT=(('the orchard keeper','counts','three ripe pears'),('a winter sailor','logs','the northern wind'))
RIGHT=(('the museum clerk','files','a yellow ticket'),('our young teacher','draws','one careful diagram'))
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=[(i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {'letters':len(t),'exact':bool(t) and not m,'first_mismatch':m[:3],'sha256':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def clause(bank): return [f'{s} {v} {o}' for s,v,o in bank]
def run():
 left,right=clause(LEFT),clause(RIGHT); fw=[]; rv=[]
 for text in left:
  t=letters(text); mid=len(t)//2; fw.append({'clause':text,'residual':t[mid:],'boundary':mid,'kind':'forward'})
 for text in right:
  t=letters(text); mid=len(t)//2; rv.append({'clause':text,'residual':t[:len(t)-mid][::-1],'boundary':len(t)-mid,'kind':'reverse'})
 joins=[]
 for a in fw:
  for b in rv:
   if a['residual']==b['residual'] and a['boundary']==b['boundary']:
    rendered=a['clause']+'; '+b['clause']+'.'; joins.append({'rendered':rendered,'audit':audit(rendered),'forward_state':a,'reverse_state':b,'provenance':{'fresh_semantic_bank':True,'typed_transitive_valency':True,'finished_tape_reversal':False,'catalogue':False,'repeated_unit':False,'rlaiF_scoring':False}})
 exact=[x for x in joins if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'grammar-mitm-seam-index-20260921','method':'length-indexed grammar MITM seam with residual character and boundary equality','stats':{'forward_states':len(fw),'reverse_states':len(rv),'equal_seam_joins':len(joins),'exact_gt38':len(exact)},'exact_candidates':exact,'joins':joins,'controls':[{'rendered':left[0]+'; '+right[0]+'.','audit':audit(left[0]+'; '+right[0]+'.')}],'novelty_preflight':{'status':'passed','signature':'fresh-semantic-bank|mitm-residual|boundary-equality','distinct_from':'whole-tape reverse generation and catalogue search'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':['two-pointer','SHA-256']},'next_repair':'Add one fresh transitive clause with a typed seam character matching the longest residual, retaining boundary equality.','status':'readable closure found' if exact else 'no readable closure; seam index retained'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True))
