"""Pre-closure lexical seam search; never mirrors a completed candidate."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/preclosure-lexical-seam-20260917.json'
SLOTS=[('quiet fox watches dawn','dawn watcher sees fox quiet'),('small child carries red kite','kite bearer carries child small')]
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); w=re.findall('[a-z]+',s.lower()); mm=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:mm.append((i,j,t[i],t[j]))
  i+=1;j-=1
 return {'text':s,'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest(),'mismatches':len(mm),'first_mismatches':mm[:4],'repeated_word':len(w)!=len(set(w)),'complete':bool(re.search('[.!?]$',s)),'reader_eligible':False}
def search(left,right):
 # obligations are consumed while selecting each word; a word pair is accepted
 # only if every newly exposed character agrees with its live counterpart.
 a=norm(left); b=norm(right); tape=[]; obligations=[]
 for x,y in zip(a,b[::-1]):
  if x!=y:return None,{'reason':'lexical seam conflict','left_char':x,'right_char':y,'obligations_live':True}
  tape.append(x); obligations.append((x,y))
 if len(a)!=len(b):return None,{'reason':'length obligation conflict','obligations_live':True}
 return left+'. '+right+'.',{'obligations_live':False,'paired_chars':len(tape)}
def main():
 rows=[]
 for i,(l,r) in enumerate(SLOTS):
  out,trace=search(l,r); rows.append({'id':i+1,'left_slot':l,'right_slot':r,'rendered':out,'trace':trace,'provenance':'hand-authored-complete-slots|preclosure-live-obligation-search','audit':audit(out) if out else None})
 result={'experiment':'preclosure-lexical-seam-20260917','operator':'incremental lexical pairing before closure','shortcut_gates':['no finished-tape mirroring','no catalogue reuse','no repeated words'],'rows':rows,'eligible_count':0}
 OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'eligible_count':0,'output':str(OUT)}))
if __name__=='__main__':main()
