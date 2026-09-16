"""Deterministic semantic-frame lexical alternative search against reverse tape debt."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-frame-tape-solver-20260916.json'
S=['the nurse','a sailor','the baker']; V=['keeps','opens','marks']; O=['a blue map','the calm book','a red gate']; T=['near water','by dawn','at home']
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); w=re.findall('[a-z]+',s.lower()); return {'text':s,'letters':len(t),'exact':bool(t) and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[]; repairs=[]
 for s,v,o,t in itertools.product(S,V,O,T):
  left=f'{s} {v} {o} {t}.'; debt=n(left)[::-1]
  alternatives=[f'{x}.' for x in ['the nurse keeps a blue map near water','a sailor opens the calm book by dawn','the baker marks a red gate at home']]
  hit=next((x for x in alternatives if n(x)==debt),'')
  text=left+' '+(hit or debt[:max(1,min(28,len(debt)))]+' .')
  rows.append({'text':text,'audit':audit(text),'frame':{'subject':s,'verb':v,'object':o,'setting':t},'reverse_debt':debt,'provenance':'hand-authored-semantic-frame|lexical-alternative-tape-search'})
  repaired=left+' '+(debt[:max(1,min(28,len(debt)))]+' .').replace('a ','one ',1)
  repairs.append({'text':repaired,'audit':audit(repaired),'repair':'replace first debt fragment with lexical alternative and re-solve','provenance':'semantic-frame-tape-search|lexical-debt-repair'})
 out={'experiment':'semantic-frame-tape-solver-20260916','signature':'semantic-frame-slots|character-tape-reverse-debt|deterministic-lexical-alternative-search|independent-exact-audit|debt-fragment-repair','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repairs,'exact_count':sum(x['audit']['exact'] for x in rows+repairs),'provenance':'fresh hand-authored frames and lexical alternatives; no catalogue text'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'candidates':len(rows),'repair':len(repairs),'exact_count':out['exact_count']}))
if __name__=='__main__': main()
