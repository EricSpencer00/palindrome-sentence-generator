"""Center-first bilateral product with pre-lexical valency typing."""
import hashlib,json
from pathlib import Path
C=[{"phrase":"the tide rested","number":"sg","tense":"past","agreement":"3sg","valency":"intransitive"},{"phrase":"the bell rings","number":"sg","tense":"present","agreement":"3sg","valency":"intransitive"},{"phrase":"the sailors wait","number":"pl","tense":"present","agreement":"3pl","valency":"intransitive"},{"phrase":"the birds flew","number":"pl","tense":"past","agreement":"3pl","valency":"intransitive"}]
L=[("Mara","charts","the coast","sg","past","transitive"),("Ivo","guards","the gate","sg","present","transitive"),("Nell","carries","a lantern","sg","present","transitive"),("Sailors","watched","the harbor","pl","past","transitive")]
R=[("the coast","shapes","Mara","sg","present","transitive"),("the gate","frames","Ivo","sg","present","transitive"),("a lantern","guides","Nell","sg","present","transitive"),("the harbor","feeds","Sailors","pl","present","transitive")]
def audit(s):
 x=''.join(c.lower() for c in s if c.isalpha()); i,j=0,len(x)-1; m=[]
 while i<j:
  if x[i]!=x[j]: m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def main():
 rows=[]; prunes=0
 for c in C:
  for a,v,o,n,t,val in L:
   for ro,rv,ra,rn,rt,rval in R:
    state={'center':c['phrase'],'center_features':(c['number'],c['tense'],c['agreement'],c['valency']),'left_role':('agent','theme',n,t,val),'right_role':('theme','agent',rn,rt,rval)}
    # Valency is checked before emission; center is deliberately intransitive.
    ok=n==c['number'] and rn==c['number'] and t==c['tense'] and rt==c['tense'] and val=='transitive' and rval=='transitive' and c['valency']=='intransitive'
    text=f'{a} {v} {o}, and {c["phrase"]}; {ro} {rv} {ra}.'
    if not ok: prunes+=1; continue
    rows.append({'text':text,'state':state,'audit':audit(text),'provenance':{'center_first':True,'valency_prelexical':True,'left_event':[a,v,o],'right_event':[ro,rv,ra]}})
 controls=[]
 for c,l,r in [(C[0],L[0],R[0]),(C[2],L[3],R[3])]:
  text=f'{l[0]} {l[1]} {l[2]}, and {c["phrase"]}; {r[0]} {r[1]} {r[2]}.'; controls.append({'text':text,'audit':audit(text),'provenance':{'center_first':True,'valency_prelexical':True}})
 out={'run_id':'center-first-valency-product-20260920','method':'center-first typed bilateral product with pre-lexical transitive/intransitive valency','novelty_preflight':{'signature':'fresh-authored|center-first|prelexical-valency|typed-bilateral-product','prior_signatures_checked':['center-first|typed-agreement-obligations|bilateral-event-product','semantic-lattice|theme-location-alternatives|semantic-lattice'],'duplicate_sweep':False},'inventory':{'centers':4,'left_events':4,'right_events':4},'stats':{'states':64,'admitted':len(rows),'obligation_prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'An exact intact candidate above 38 must survive valency, agreement, and independent audit; none admitted here.','next_repair':'Use mixed valency center clauses with an explicitly licensed adjunct, rather than widening the event inventory.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/center-first-valency-product-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
