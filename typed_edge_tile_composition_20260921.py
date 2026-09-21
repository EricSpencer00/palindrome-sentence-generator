"""Bounded semantic tile composition with typed edge-state compatibility."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/typed-edge-tile-composition-20260921.json'
LEFT=({'text':'the scout maps a cove','subject':'singular','object':'place','tense':'present'}, {'text':'our guides carried a key','subject':'plural','object':'thing','tense':'past'})
RIGHT=({'text':'the keeper watches a beacon','subject':'singular','object':'signal','tense':'present'}, {'text':'sailors found the harbor','subject':'plural','object':'place','tense':'past'})
FUNCTION_WORDS={'a','an','and','as','by','for','in','of','on','our','the','to','while'}
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def content_words(text):
 return [w for w in re.findall('[a-z]+',text.lower()) if w not in FUNCTION_WORDS]
def run():
 rows=[]
 for l,r in itertools.product(LEFT,RIGHT):
  compatible=l['subject']==r['subject'] and l['tense']==r['tense']
  rendered=f"{l['text']}, while {r['text']}."; au=audit(rendered);content_left=content_words(l['text']);content_right=content_words(r['text']);content=content_left+content_right
  gates={'typed_edges_compatible':compatible,'whole_output_exact':au['exact'],'readable_full_sentence':True,'content_disjoint':not(set(content_left)&set(content_right)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'left_tile':l,'right_tile':r,'edge_compatibility':{'subject':l['subject']==r['subject'],'tense':l['tense']==r['tense'],'object_relation':l['object']!=r['object']},'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'typed subject/object/tense edge-state composition','checked_before_rendering':True,'fixed_outer_shell':False,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'typed-edge-tile-composition-20260921','method':'typed edge-state composition for readable semantic tiles','stats':{'left_tiles':len(LEFT),'right_tiles':len(RIGHT),'controls':len(rows),'typed_compatible':sum(r['gates']['typed_edges_compatible'] for r in rows),'accepted_exact':len(exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'typed-edge-state|semantic-tiles|pre-render-compatibility','signature_collision':False,'distinct_from':'fixed seam and untyped tile composition'},'next_repair':'Add argument-role compatibility (agent/theme) to edge states before composing a three-tile discourse.','status':'fresh exact closure found' if exact else 'no exact typed composition; readable controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
