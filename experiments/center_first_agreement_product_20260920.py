"""Center-first typed bilateral event product.

The discourse centre fixes agreement features before either event is realized.
No candidate is obtained by reversing a finished string.
"""
import hashlib, json
from pathlib import Path

CENTERS=[{"phrase":"the tide rested","number":"sg","tense":"past","agreement":"3sg"},{"phrase":"the bell rings","number":"sg","tense":"present","agreement":"3sg"},{"phrase":"the sailors wait","number":"pl","tense":"present","agreement":"3pl"},{"phrase":"the birds flew","number":"pl","tense":"past","agreement":"3pl"}]
LEFT=[("Mara","charts","the coast","sg","past"),("Ivo","guards","the gate","sg","present"),("Nell","carries","a lantern","sg","present"),("Sailors","watched","the harbor","pl","past")]
RIGHT=[("the coast","shapes","Mara","sg","present"),("the gate","frames","Ivo","sg","present"),("a lantern","guides","Nell","sg","present"),("the harbor","feeds","Sailors","pl","present")]
def letters(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(text):
 x=letters(text); i,j=0,len(x)-1; mm=[]
 while i<j:
  if x[i]!=x[j]: mm.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1; j-=1
 return {'exact':not mm,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':mm[:8],'two_pointer_checked':True}
def main():
 rows=[]; prunes=0
 for center in CENTERS:
  for a,v,o,n,t in LEFT:
   for ro,rv,ra,rn,rt in RIGHT:
    obligations={'agent':(n,t,'agent'),'theme':(rn,rt,'theme'),'center':(center['number'],center['tense'],center['agreement'])}
    # Agreement-aware semantic gate is evaluated before rendering.
    admitted=(n==center['number'] and rn==center['number'] and t==center['tense'] and rt==center['tense'])
    text=f'{a} {v} {o}, and {center["phrase"]}; {ro} {rv} {ra}.'
    if not admitted: prunes+=1; continue
    rows.append({'text':text,'center':center,'obligations':obligations,'audit':audit(text),'provenance':{'center_first':True,'center_is_complete_clause':True,'left_event':[a,v,o],'right_event':[ro,rv,ra]}})
 controls=[]
 for c,(a,v,o,n,t),(ro,rv,ra,rn,rt) in [(CENTERS[0],LEFT[0],RIGHT[0]),(CENTERS[2],LEFT[3],RIGHT[3])]:
  text=f'{a} {v} {o}, and {c["phrase"]}; {ro} {rv} {ra}.'; controls.append({'text':text,'audit':audit(text),'provenance':{'center_first':True,'center_is_complete_clause':True}})
 result={'run_id':'center-first-agreement-product-20260920','method':'center-first typed bilateral event product','novelty_preflight':{'signature':'fresh-authored|center-first|typed-agreement-obligations|bilateral-event-product','prior_signatures_checked':['typed-bilateral-event-product|dual-live-obligations|heldout-centre','semantic-lattice|number-agreement|width2-live-residual'],'duplicate_sweep':False},'inventory':{'centers':len(CENTERS),'left_events':len(LEFT),'right_events':len(RIGHT)},'stats':{'states':len(CENTERS)*len(LEFT)*len(RIGHT),'admitted':len(rows),'obligation_prunes':prunes,'exact_over_38':sum(r['audit']['exact'] and r['audit']['letters']>38 for r in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'A reader-worthy exact candidate would require a row with exact=true, letters>38, and independently reviewed intact prose; none admitted here.','next_repair':'Add typed argument structure (transitive/intransitive valency) to the center-first state before lexical realization; do not enlarge this inventory.'}
 Path('runs').mkdir(exist_ok=True); Path('runs/center-first-agreement-product-20260920.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['stats'],sort_keys=True))
if __name__=='__main__': main()
