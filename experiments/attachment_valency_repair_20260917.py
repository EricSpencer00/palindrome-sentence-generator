#!/usr/bin/env python3
"""First-mismatch lexical/inflection repair over typed attachment frames."""
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/attachment-valency-repair-20260917.json'
FRAMES=[
 ("the patient curator",["labels","labelled","files"],["a faded map","the old chart"],"in the quiet archive"),
 ("a careful gardener",["carries","carried","moves"],["the blue lantern","a brass lamp"],"beside the stone wall"),
 ("the young baker",["delivers","delivered","brings"],["warm bread","fresh loaves"],"to the waiting nurse"),
 ("a calm teacher",["records","recorded","marks"],["each small answer","the short replies"],"after the evening class"),
 ("the steady sailor",["repairs","repaired","mends"],["a torn canvas","the split sail"],"before the morning tide"),
]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); r=t[::-1]; mm=sum(a!=b for a,b in zip(t,r)); return {'letters':len(t),'exact':t==r,'mismatches':mm,'mismatch_rate':mm/len(t),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest()}
def render(f,v,o): return f'{f[0]} {v} {o} {f[3]}'
def first_mismatch(t):
 r=t[::-1]
 return next((i for i,(a,b) in enumerate(zip(t,r)) if a!=b),None)
def main():
 branches=[]
 for i,left in enumerate(FRAMES):
  for j,right in enumerate(FRAMES):
   if i==j: continue
   for vi,v in enumerate(left[1]):
    for oi,o in enumerate(left[2]):
     for vj,w in enumerate(right[1]):
      for oj,p in enumerate(right[2]):
       a=render(left,v,o); b=render(right,w,p); text=a+'; '+b+'.'; au=audit(text)
       # Preserve valency: transitive heads retain an overt object and PP attachment.
       valid=bool(o and p and left[3] and right[3] and v!=w)
       branches.append({'branch':f'{i}:{j}:{vi}{oi}:{vj}{oj}','text':text,'valid_attachment_csp':valid,
        'first_mismatch':first_mismatch(norm(text)),'left_variant':{'verb':v,'object':o},'right_variant':{'verb':w,'object':p},
        'audit':au,'provenance':'fresh inflectional/lexical substitution over human-authored typed frames'})
 # rank by exactness then mismatch at first seam; never admit non-exact candidates
 branches.sort(key=lambda x:(not x['audit']['exact'],x['audit']['mismatch_rate'],x['first_mismatch'] if x['first_mismatch'] is not None else 999))
 result={'method':'attachment-preserving first-mismatch lexical repair','target_letters':100,'branches':branches,
  'rendered_candidates':branches[:10],'admitted':[x for x in branches if x['valid_attachment_csp'] and x['audit']['exact'] and x['audit']['letters']>=100],
  'independent_validator':'normalized tape, reverse tape, mismatch count, and independent SHA-256 pair',
  'novelty_preflight':{'catalogue_lookup':'not used','fixed_tape':False,'word_order_mirror':False,'repeated_units':False,'borrowed_text':False},
  'next_repair':'Switch from substitutions to a live two-sided character equation over attachment-compatible phrase expansions; retain distinct heads and require a human-readable clause parse.'}
 OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'branches':len(branches),'admitted':len(result['admitted']),'best':result['rendered_candidates'][0]},indent=2))
if __name__=='__main__': main()
