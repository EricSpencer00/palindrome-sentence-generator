import hashlib,json,re
from pathlib import Path
W=re.compile('[a-z]+'); ROOT=Path(__file__).resolve().parents[1]
LEFT=("A careful pilot maps rivers at dawn","The patient keeper guards old charts","A quiet singer carries small parcels","The gardener marks western roads")
RIGHT=("A baker records bright letters at noon","Some guards watch cedar fences with care","The sailor reads quiet rivers after rain","A nurse carries old parcels near home")
def n(s): return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s); i=0; j=len(t)-1
 while i<j and t[i]==t[j]: i+=1; j-=1
 return {'exact':i>=j,'first_mismatch':None if i>=j else [i,j,t[i],t[j]],'letters':len(t),'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'two_pointer_exact':i>=j}
rows=[]
for l in LEFT:
 for r in RIGHT:
  s=l+'; a; '+r; a=audit(s)
  if a['letters']>38: rows.append({'text':s,'audit':a,'provenance':{'manual_authored':True,'catalogue':False,'anchor_wrap':False,'repeated':False,'self_palindrome':False}})
out={'experiment':'manual-clause-seam-20260919','rows':rows,'exact':sum(x['audit']['exact'] for x in rows),'next_discriminator':'author a subject/object seam pair indexed by the recorded boundary letters'}
(ROOT/'runs').mkdir(exist_ok=True); (ROOT/'runs/manual-clause-seam-20260919.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'exact':out['exact'],'first':rows[0]},indent=2))
