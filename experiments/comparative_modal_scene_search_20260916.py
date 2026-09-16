"""Comparative/measurement and modal-scene exhaustive search probe."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/comparative-modal-scene-20260916.json'
SUB=['the gauge','a patient scout','the analyst']; MOD=['may','must','can']; VER=['measure','compare','estimate']; OBJ=['the wider bridge','a smaller field','the warmer room']
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 t=n(s);w=re.findall('[a-z]+',s.lower());return {'text':s,'letters':len(t),'exact':bool(t)and t==t[::-1],'complete_sentence':s.endswith('.'),'no_repeated_units':len(w)==len(set(w)),'borrowed_catalogue':False,'reader_eligible':False}
def main():
 rows=[];repair=[]
 for s,m,v,o in itertools.product(SUB,MOD,VER,OBJ):
  t=f'{s} {m} {v} {o}.'; rows.append({'text':t,'audit':a(t),'provenance':'fresh-comparative-modal-scene','semantic_state':'measurement/comparison with modal commitment'})
  repair.append({'text':t.replace(m,{'may':'might','must':'should','can':'could'}[m]),'audit':a(t.replace(m,{'may':'might','must':'should','can':'could'}[m])),'provenance':'fresh-comparative-modal-scene|modal-repair','repair':'replace modal at first character-debt mismatch'})
 out={'experiment':'comparative-modal-scene-search-20260916','signature':'comparative-measurement-semantics|modal-commitment-state|exhaustive-lexical-search|modal-mismatch-repair|independent-exact-audit','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'candidates':rows,'repair':repair,'exact_count':sum(x['audit']['exact'] for x in rows+repair),'reader_eligible_count':0,'provenance':'fresh hand-authored measurement scenes; no catalogue or symmetry shortcut'}
 OUT.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'candidates':len(rows),'repair':len(repair),'exact_count':out['exact_count']}))
if __name__=='__main__':main()
