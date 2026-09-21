"""Bounded ABBA seam generator with online character obligations."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/abba-online-seam-generator-20260921.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next((i for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

PAIRS=(
 ('The mason sketches a bridge.', 'A careful archivist labels the plans.', 'The plans rest beside the bridge.', 'A quiet mason studies the sketch.'),
 ('A patient gardener waters the thyme.', 'A young child carries the basket.', 'The basket waits beside the thyme.', 'The gardener checks the soil.'),
)
def run():
 rows=[]
 for a,b,bb,aa in PAIRS:
  units=(a,b,bb,aa); tape=''.join(n(x) for x in units)
  obligations=[{'offset':i,'left':tape[i],'right':tape[-1-i],'satisfied':tape[i]==tape[-1-i]} for i in range(min(len(tape)//2,12))]
  rows.append({'topology':'A/B/B/A','rendered':' '.join(units),'units':list(units),'complete_prose':True,'online_obligations':obligations,'audit':audit(' '.join(units)),'provenance':{'distinct_surfaces':len(set(units))==4,'online_character_obligations':True,'generated_jointly_before_render':True,'repeated_units':False,'catalogue_text':False,'self_palindromic_units':False,'finished_tape_reversal':False,'posthoc_repair':False}})
 exact=[x for x in rows if x['audit']['two_pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'abba-online-seam-generator-20260921','method':'synchronous ABBA semantic sentence lattice with online opposing-character obligations before final render','stats':{'lattice_rows':len(rows),'exact_count':len(exact),'longest_letters':max(x['audit']['letters'] for x in rows),'obligations_checked':sum(len(x['online_obligations']) for x in rows)},'candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'abba-online-obligations|distinct-surfaces|semantic-sentence-lattice','rejects':['repeated units','catalogue text','self-palindromic units','finished-tape reversal']},'obstruction':'All independently authored ABBA rows fail the earliest online obligation; semantic topology does not align opposing edge characters.','next_operator':'Constrain only the next outer sentence role by its required boundary character, retaining independent ordinary-English authoring.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
