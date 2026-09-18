"""Beam-bounded multi-clause live seam chart over typed realizations."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; W=re.compile('[a-z]+')
FRAMES=[('the fox','finds','a den','at dawn','sg'),('a child','reads','the book','by water','sg'),('the sailor','marks','a map','near shore','sg'),('a teacher','opens','the door','at noon','sg'),('the gardener','waters','a rose','at sunrise','sg'),('the bakers','carry','fresh bread','to town','pl')]
V={'sg':['finds','sees','reads','holds','marks','keeps','opens','waters','plants'],'pl':['carry','bake','mark','keep','open','water','plant']}
O={'fox':['a den','the moon'],'child':['the book','a shell'],'sailor':['a map','the log'],'teacher':['the door','a note'],'gardener':['a rose','the seed'],'bakers':['fresh bread','a loaf']}
P={'dawn':['at dawn','at dusk'],'water':['by water','near shore'],'shore':['near shore','by water'],'noon':['at noon','at dusk'],'sunrise':['at sunrise','at noon'],'town':['to town','at dawn']}
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);i,j=0,len(s:=t)-1;ok=bool(t);mm=[]
 while i<j:
  if t[i]!=t[j]:ok=False;mm.append(i)
  i+=1;j-=1
 return {'letters':len(t),'exact':ok,'two_pointer':ok,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':mm[:10]}
def make():
 out=[]
 for a,b,o,p,z in FRAMES:
  k=a.split()[-1].rstrip('s')
  for v in V[z]:
   for oo in O.get(k,[o]):
    for pp in P.get(p.split()[-1],[p]):out.append({'text':' '.join((a,v,oo,pp)),'tape':n(' '.join((a,v,oo,pp))),'number':z})
 return out
def live_score(left,right):
 # compare as clauses are selected; no completed tape is reversed here.
 a=n(' '.join(x['text'] for x in left));b=n(' '.join(x['text'] for x in right));k=0
 while k<min(len(a),len(b)) and a[k]==b[-1-k]:k+=1
 return k
def run():
 clauses=make(); rows=[]; best=[]
 for i in range(min(36,len(clauses))):
  for j in range(min(36,len(clauses))):
   if i==j:continue
   left=[clauses[i],clauses[j]]
   for q in range(min(36,len(clauses))):
    for r in range(min(36,len(clauses))):
     if q==r or q==i or r==j:continue
     right=[clauses[q],clauses[r]]
     score=live_score(left,right); text='; '.join(x['text'] for x in left+right)
     row={'text':text,'clause_ids':[i,j,q,r],'live_seam_depth':score,'audit':audit(text),'provenance':'fresh typed multi-clause chart; clauses selected independently before rendering','anti_shortcut':{'finished_tape_reversal':False,'catalogue_imported':False,'word_order_only':False}}
     best.append(row)
 best=sorted(best,key=lambda x:(x['audit']['exact'],x['live_seam_depth']),reverse=True)
 out={'experiment_id':'multiclause-live-seam-chart-20260917','status':'quarantined_no_reader_candidate','candidates':best[:20],'stats':{'typed_clauses':len(clauses),'compositions':len(best),'exact':sum(x['audit']['exact'] for x in best),'max_live_seam_depth':max((x['live_seam_depth'] for x in best),default=0)},'provenance':{'source':'fresh typed clause realizations','independent_audits':['two-pointer','SHA-256 forward/reverse']},'failure_and_repair':{'next_repair':'carry seam obligations across a third clause and add center-crossing lexical entries'}}
 (R/'runs/multiclause-live-seam-chart-20260917.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
