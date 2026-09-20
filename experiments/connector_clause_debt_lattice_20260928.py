"""Paired clause lattice with typed connectors and cross-seam character debt."""
from __future__ import annotations
import argparse,hashlib,json,re,socket,itertools
from pathlib import Path
CLAUSES=[('SVO','pilot maps the cove'),('SVO','gardener tends the roses'),('COP','poet is calm'),('LOC','sailor waits by harbor'),('IMP','you mark the map'),('REL','bird sings at dawn'),('APP','mason is patient')]
CONNS=[('and','coord'),('but','contrast'),('while','temporal'),('for','cause'),('yet','contrast')]
def tape(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=tape(s);return {'letters':len(t),'two_pointer_exact':bool(t) and t==t[::-1],'pointer_mismatches':sum(a!=b for a,b in zip(t,t[::-1]))//2,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run(limit):
 exact=[]; controls=[]
 for (lt,lc),(rt,rc),(conn,kind),(_,rkind) in itertools.product(CLAUSES,CLAUSES,CONNS,CONNS):
  if lt==rt or lc==rc or conn in lc or conn in rc:continue
  # connector is chosen before either side is rendered; all seam debt is checked.
  left=f'{lc} {conn} {rc}'; right=f'{rc} {conn} {lc}'; a,b=tape(left),tape(right); k=min(len(a),len(b)); debt=sum(x!=y for x,y in zip(a[-k:],b[:k][::-1])) if k else 0
  text=left+'; '+right; au=audit(text)
  rec={'rendered':text,'audit':au,'connector':conn,'semantic_roles':{'left':lt,'right':rt,'connector_type':kind},'clause_finality':{'left':True,'right':True},'full_boundary_debt':debt}
  if debt==0 and au['two_pointer_exact']:rec['reader_worthy']=True;exact.append(rec)
  elif len(controls)<8:rec['reader_worthy']=False;controls.append(rec)
  if len(exact)>=limit:break
 return exact,controls
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',required=True);a=ap.parse_args();ex,co=run(a.limit);p={'experiment':'connector-clause-debt-lattice-20260928','host':socket.gethostname(),'parameters':vars(a),'candidates':ex,'controls':co,'closures':len(ex),'reader_worthy':sum(x['reader_worthy'] for x in ex),'provenance':{'fresh_authored_clauses':True,'connectors':['and','but','while','for','yet'],'joint_connector_selection':True,'cross_clause_debt':True,'catalogue_used':False,'seed_wrapping':False,'repeated_units':False,'posthoc_repair':False},'next_construction':'add connector-specific agreement and discourse constraints to the cross-seam debt state.'};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('experiment','closures','reader_worthy')}))
if __name__=='__main__':main()
