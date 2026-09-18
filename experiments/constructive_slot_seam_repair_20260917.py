"""Constructive lane: LLM proposes clause slots; deterministic tape repair audits output."""
import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/constructive-slot-seam-repair-20260917.json'; MODEL='gpt-oss:20b'
REQ='Give three ordinary, complete English clause slots about one concrete scene; 8-14 words each, distinct words, no palindromes, no quotations. Output clauses only.'
def n(s): return re.sub('[^a-z]','',s.lower())
def mism(t): return [(i,len(t)-1-i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
def repair(s):
 c=list(n(s)); i,j=0,len(c)-1
 while i<j:c[j]=c[i];i+=1;j-=1
 return ''.join(c)
def audit(s,cat):
 t=n(s); w=re.findall('[A-Za-z]+',s); mm=mism(t)
 complete=bool(re.search('[.!?]$',s.strip())) and len(w)>=8
 return {'text':s,'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest(),'mismatches':len(mm),'first_mismatches':mm[:5],'repeated_word':len({x.lower() for x in w})<len(w),'catalogue_match':t in cat,'complete':complete,'reader_eligible':len(t)>=100 and complete and not mm and not (len({x.lower() for x in w})<len(w)) and t not in cat}
def ask():
 try:return subprocess.run(['ollama','run',MODEL,REQ],capture_output=True,text=True,timeout=25).stdout.strip(),None
 except Exception as e:return '',f'{type(e).__name__}: {e}'
def main():
 k=json.loads((ROOT/'data/known_palindromes.json').read_text()); cat=set(k if isinstance(k,list) else k.keys()); rows=[]
 for i in range(3):
  raw,err=ask(); clauses=[x.strip() for x in raw.splitlines() if x.strip()][:3]; seed=' '.join(clauses); tape=repair(seed) if seed else ''
  rows.append({'id':f'slot-{i+1}','request':REQ,'provenance':'ollama-slot-proposal|deterministic-seam-repair','raw_slots':clauses,'rendered_candidate':tape,'audit':audit(tape,cat) if tape else None,'error':err})
 out={'experiment':'constructive-slot-seam-repair-20260917','model':MODEL,'attempts':rows,'repair':'mirror normalized left half onto right; quarantined unless prose gates pass','eligible_count':sum(x['audit']['reader_eligible'] for x in rows if x['audit'])}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'eligible_count':out['eligible_count'],'output':str(OUT)}))
if __name__=='__main__':main()
