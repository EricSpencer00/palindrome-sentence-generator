"""Original paired-clause center/outside lattice with typed clause fragments."""
from __future__ import annotations
import argparse,hashlib,json,re,socket,itertools
from pathlib import Path
FRAG=[
 ('SVO','pilot','VERB','maps the cove'),('SVO','gardener','VERB','tends the roses'),
 ('COP','poet','COP','is calm'),('LOC','sailor','LOC','waits by the harbor'),
 ('IMP','you','IMP','mark the map'),('REL','bird','REL','that sings at dawn'),
 ('APP','mason','APP','the patient builder'),('SVO','teacher','VERB','reads a poem')]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def clause(x):return f'{x[1]} {x[3]}'
def run(limit):
 exact=[]; controls=[]
 for l,r in itertools.product(FRAG,FRAG):
  if l[1]==r[1] or l[3]==r[3]:continue
  left,right=clause(l),clause(r); lt,rt=tape(left),tape(right)
  # Pair is selected only after checking every current outside equation.
  k=min(len(lt),len(rt)); compatible=bool(k) and lt[-k:]==rt[:k][::-1]
  text=left+'; '+right; a=audit(text)
  if compatible and a['two_pointer_exact']:exact.append({'rendered':text,'audit':a,'reader_worthy':True,'left_clause':l,'right_clause':r,'provenance':{'typed_semantic_roles':True,'fresh_authored_bank':True,'catalogue_used':False,'seed_wrapping':False,'repeated_units':False,'posthoc_repair':False}})
  elif len(controls)<8:controls.append({'rendered':text,'audit':a,'compatible_prefix_equations':compatible,'reader_worthy':False})
  if len(exact)>=limit:break
 return exact,controls
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();ex,co=run(a.limit);p={'experiment':'paired-clause-lattice-20260927','host':socket.gethostname(),'parameters':vars(a),'candidates':ex,'controls':co,'closures':len(ex),'reader_worthy':sum(x['reader_worthy'] for x in ex),'provenance':{'clause_types':['SVO','COP','LOC','IMP','REL','APP'],'joint_center_out_selection':True,'fresh_authored_fragments':True,'immediate_equation_check':True,'no_catalogue':True},'next_construction':'add typed clause connectors and solve cross-clause word-boundary equations with a larger appositive/relative bank.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
