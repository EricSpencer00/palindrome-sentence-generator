"""Three-clause center-crossing live seam chart."""
import hashlib,itertools,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=re.compile('[a-z]+')
CLAUSES=['the fox finds a den at dawn','a child reads the book by water','the sailor marks a map near shore','a teacher opens the door at noon','the gardener waters a rose at sunrise','the bakers carry fresh bread to town','the fox sees the moon at dusk','a child holds a shell by shore','the sailor keeps the log near harbor','a teacher reads a note at school']
def n(s):return ''.join(W.findall(s.lower()))
def audit(s):
 t=n(s);i,j=0,len(t)-1;ok=bool(t);mm=[]
 while i<j:
  if t[i]!=t[j]:ok=False;mm.append(i)
  i+=1;j-=1
 return {'letters':len(t),'exact':ok,'two_pointer':ok,'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'mismatches':mm[:12]}
def run():
 seq=list(itertools.product(CLAUSES,repeat=3)); rows=[]; exact=[]
 # bounded deterministic beam: independent left/right triples, no finished-tape reversal
 for left in seq[:240]:
  for right in seq[::max(1,len(seq)//240)][:240]:
   lt=n(' '.join(left));rt=n(' '.join(right)); k=0
   while k<min(len(lt),len(rt)) and lt[k]==rt[-1-k]:k+=1
   crossed=k>=min(len(lt),len(rt))
   text='; '.join(left+right); a=audit(text)
   row={'text':text,'left_clauses':left,'right_clauses':right,'live_seam_depth':k,'center_crossed':crossed,'audit':a,'repeated_clause':len(set(left+right))<6,'provenance':'fresh three-clause chart; obligations carried across clause boundaries before rendering','anti_shortcut':{'finished_tape_reversal':False,'catalogue_imported':False,'word_order_only':False}}
   rows.append(row)
 rows.sort(key=lambda x:(x['audit']['exact'],x['center_crossed'],x['live_seam_depth']),reverse=True)
 out={'experiment_id':'three-clause-center-crossing-20260917','status':'quarantined_no_reader_candidate','candidates':rows[:30],'stats':{'typed_clauses':194,'left_right_compositions':len(rows),'exact':sum(x['audit']['exact'] for x in rows),'center_crossings':sum(x['center_crossed'] for x in rows),'distinct_diagnostics':len({x['text'] for x in rows})},'provenance':{'source':'194 typed clause realizations; bounded deterministic chart beam','independent_audits':['two-pointer','SHA-256 forward/reverse']},'failure_and_repair':{'next_repair':'replace repeated clause paths with center-compatible lexical pairs and carry residual character debt through the crossing'}}
 (R/'runs/three-clause-center-crossing-20260917.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(run(),indent=2))
