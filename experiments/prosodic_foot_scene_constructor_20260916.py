"""Prosodic-foot templates coupled to complete semantic scenes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT="prosodic-foot-scene-constructor-20260916"
SIGNATURE="prosodic-foot-template|stress-rhythm-realization|semantic-role-scene|character-debt-audit|independent-exact-audit|prosodic-repair"
SCENES=[("The patient gardener","carefully waters","the young roses","beside the stone wall"),("A thoughtful teacher","quietly carries","the blue atlas","into the reading room"),("The local artist","gently frames","a bright portrait","above the wooden desk")]
FEET=["", " at dawn", " in the calm morning"]
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-i-1]]
 return {'exact':bool(t) and not bad,'letters':len(t),'mismatches':len(bad),'first_mismatch':bad[0] if bad else None}
def render(s,foot): return f"{s[0]} {s[1]} {s[2]} {s[3]}{foot}."
def run(phase,repair=False):
 feet=FEET if not repair else [" at first light"," through the quiet morning"," in gentle measured steps"]
 rows=[]
 for a in SCENES:
  for b in SCENES:
   if a==b: continue
   for fa in feet:
    for fb in feet:
     left,right=render(a,fa),render(b,fb); text=left+' '+right
     toks=re.findall('[a-z]+',text.lower())
     rows.append({'phase':phase,'feet':[fa,fb],'left':left,'right':right,'rendered':text,'audit':audit(text),'complete_sentences':True,'all_different_content_words':len(toks)==len(set(toks)),'reader_eligible':False,'provenance':'independently authored semantic scenes with generated prosodic-foot variants; no catalogue text'})
 return rows
def main():
 base,repair=run('base'),run('repair',True)
 p={'experiment':EXPERIMENT,'signature':SIGNATURE,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operator':'jointly enumerate semantic scene roles and rhythmic foot templates before character-debt audit','base':{'candidates':base,'exact_count':sum(x['audit']['exact'] for x in base)},'repair':{'candidates':repair,'exact_count':sum(x['audit']['exact'] for x in repair)},'repair_action':'replace the base foot inventory with held-out iambic-like timing phrases and rerun the same semantic scenes','provenance':{'catalogue_used':False,'borrowed_text':False,'word_order_mirror':False,'fragments':False,'repeated_units_allowed':False}}
 (ROOT/'runs/prosodic-foot-scene-constructor-20260916.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps({'base':len(base),'repair':len(repair),'base_exact':p['base']['exact_count'],'repair_exact':p['repair']['exact_count'],'max_letters':max(x['audit']['letters'] for x in repair)}))
if __name__=='__main__': main()
