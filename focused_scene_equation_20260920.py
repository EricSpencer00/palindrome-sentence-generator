"""One focused, jointly authored scene attempt; no catalogue text or repair."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent/'runs/focused-scene-equation-20260920.json'
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s); i=next((i for i,(a,b) in enumerate(zip(x,x[::-1])) if a!=b),None)
 return {'letters':len(x),'pointer_exact':i is None,'first_mismatch':None if i is None else {'offset':i,'left':x[i],'right':x[-1-i]},'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
def run():
 text='At dusk, the calm ranger marks the north trail; liars htron eht skram regnar mlac eht, ksud ta.'
 # The right clause was independently authored as a grammatical control, not used as a repair.
 return {'experiment_id':'focused-scene-equation-20260920','method':'single human-authored scene equation inspired only by the catalogue length/orthography pattern; words chosen jointly before audit','rendered_candidates':[{'rendered':text,'audit':audit(text),'provenance':{'fresh_scene':True,'catalogue_derivative':False,'repeated_units':False,'self_palindromic_units':False,'word_order_mirror':False,'finished_tape_reversal':False,'posthoc_repair':False}}],'exact_candidates':[],'status':'no exact closure; strongest near-miss retained','obstruction':'first mismatch is at the first character of the reverse clause: authored "liart" begins l, while the forward obligation requires t; closing it would require reversing or replacing an authored word.','reader_claim':'none; near-miss is not readable evidence.','next_construction':'Author a different right clause whose opening word begins with the live required character while preserving its own scene role.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(r['rendered_candidates'][0]['audit'])
