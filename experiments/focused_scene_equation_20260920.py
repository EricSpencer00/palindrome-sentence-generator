"""One focused, jointly authored scene attempt; no catalogue text or repair."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/focused-scene-equation-20260920.json'
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s); i=next((i for i,(a,b) in enumerate(zip(x,x[::-1])) if a!=b),None)
 return {'letters':len(x),'pointer_exact':i is None,'first_mismatch':None if i is None else {'offset':i,'left':x[i],'right':x[-1-i]},'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
def run():
 text='At dusk, the calm ranger marks the north trail; liars htron eht skram regnar mlac eht, ksud ta.'
 # This is deliberately retained as an orthographic diagnostic only.  The
 # right-side words are visibly reversed fragments, so it is not an authored
 # prose candidate and must never enter the reader gate.
 return {'experiment_id':'focused-scene-equation-20260920','method':'single orthographic near-miss diagnostic inspired only by catalogue length/orthography; not a prose candidate','rendered_candidates':[{'rendered':text,'audit':audit(text),'provenance':{'diagnostic_only':True,'fresh_scene':False,'right_side_reversed_fragments':True,'catalogue_derivative':False,'repeated_units':False,'self_palindromic_units':False,'word_order_mirror':False,'finished_tape_reversal':True,'posthoc_repair':False}}],'exact_candidates':[],'status':'rejected diagnostic; right side is reversed fragment text','obstruction':'first mismatch is at offset 32; the apparent right clause is not grammatical and is excluded before any readability or exactness claim.','reader_claim':'none; rejected gibberish diagnostic.','next_construction':'Return to a construction that emits both sides in ordinary grammatical order; do not reverse a finished clause.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(r['rendered_candidates'][0]['audit'])
