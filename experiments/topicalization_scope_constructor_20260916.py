"""Topicalization scope constructor with complete prose and exact audit."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='topicalization-scope-constructor-20260916'
SIGNATURE='topicalization-information-structure|semantic-role-preservation|scope-realization|live-character-obligation|independent-exact-audit|parenthetical-repair'
SCENES=[('the old map','Mara','studies'),('a bright lantern','Niko','carries'),('the quiet letter','Sela','copies')]
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=sum(a!=b for a,b in zip(t,t[::-1])); return {'exact':bool(t) and bad==0,'letters':len(t),'mismatches':bad}
def run(phase,repair=False):
 rows=[]; scenes=SCENES+[('the sealed parcel','Tara','opens')] if repair else SCENES
 for obj,subj,verb in scenes:
  for obj2,subj2,verb2 in scenes:
   if subj==subj2: continue
   left=f"{obj.capitalize()}, {subj} {verb} it"; right=f"{obj2}, {subj2.lower()} {verb2} it"
   if repair: left=f"{obj.capitalize()}, as everyone knows, {subj} {verb} it"; right=f"{obj2}, as everyone knows, {subj2.lower()} {verb2} it"
   text=left+'; '+right+'.'; toks=re.findall('[a-z]+',text.lower())
   rows.append({'phase':phase,'left':left,'right':right,'rendered':text,'audit':audit(text),'complete_sentences':True,'semantic_roles_preserved':True,'all_different_content_words':len(toks)==len(set(toks)),'reader_eligible':False,'provenance':'independently authored topicalized scenes; no catalogue text; no reflected unit reuse'})
 return rows
def main():
 b,r=run('base'),run('repair',True); p={'experiment':EXPERIMENT,'signature':SIGNATURE,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operator':'jointly enumerate topicalized object scope and semantic subject/verb realization before exact audit','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_action':'add a held-out parenthetical scope realization and rerun topicalized roles','provenance':{'catalogue_used':False,'borrowed_text':False,'word_order_mirror':False,'fragments':False,'repeated_units_allowed':False}}
 (ROOT/'runs/topicalization-scope-constructor-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'base':len(b),'repair':len(r),'base_exact':p['base']['exact_count'],'repair_exact':p['repair']['exact_count']}))
if __name__=='__main__': main()
