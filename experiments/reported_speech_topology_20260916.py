"""Reported-speech topology coupled to semantic attribution and exact audit."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='reported-speech-topology-20260916'
SIGNATURE='reported-speech-topology|semantic-attribution-embedding|tense-shifted-content|live-character-obligation|independent-exact-audit|conditional-repair'
SCENES=[('Mara','said','the keeper opened the gate'),('Niko','reported','the sailor found a map'),('Sela','claimed','the doctor carried a note')]
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=sum(a!=b for a,b in zip(t,t[::-1])); return {'exact':bool(t) and not bad,'letters':len(t),'mismatches':bad}
def run(phase,repair=False):
 rows=[]; scenes=SCENES+[('Tara','explained','the mason repaired the wall')] if repair else SCENES
 for a in scenes:
  for b in scenes:
   if a==b: continue
   left=f"{a[0]} {a[1]} that {a[2]}"; right=f"{b[0].lower()} {b[1]} that {b[2]}"
   if repair: left=f"{a[0]} {a[1]} that, if true, {a[2]}"; right=f"{b[0].lower()} {b[1]} that, if true, {b[2]}"
   text=left+'; '+right+'.'; toks=re.findall('[a-z]+',text.lower())
   rows.append({'phase':phase,'left':left,'right':right,'rendered':text,'audit':audit(text),'complete_sentences':True,'all_different_content_words':len(toks)==len(set(toks)),'reader_eligible':False,'provenance':'independently authored reported-speech scenes; no catalogue text; no reflected unit reuse'})
 return rows
def main():
 base,repair=run('base'),run('repair',True); p={'experiment':EXPERIMENT,'signature':SIGNATURE,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operator':'jointly enumerate attribution and embedded proposition topology before exact character audit','base':{'candidates':base,'exact_count':sum(x['audit']['exact'] for x in base)},'repair':{'candidates':repair,'exact_count':sum(x['audit']['exact'] for x in repair)},'repair_action':'insert a conditional subordinate topology into both attributed propositions and rerun with a held-out speaker/event','provenance':{'catalogue_used':False,'borrowed_text':False,'word_order_mirror':False,'fragments':False,'repeated_units_allowed':False}}
 (ROOT/'runs/reported-speech-topology-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'base':len(base),'repair':len(repair),'base_exact':p['base']['exact_count'],'repair_exact':p['repair']['exact_count'],'max_letters':max(x['audit']['letters'] for x in repair)}))
if __name__=='__main__': main()
