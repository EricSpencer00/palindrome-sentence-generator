"""Semordnilap graph index used only to choose fresh clause vocabulary."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parents[1]/'runs/semordnilap-phrase-graph-constructor-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
DETS={'a','an','the','our','my','no'}
def role(phrase):
 words=phrase.split()
 if len(words)!=2:return 'fragment'
 if words[0] in DETS and words[1].isalpha():
  if words[0]=='a' and words[1][0] in 'aeiou': return 'NP_INVALID_ARTICLE'
  if words[0]=='an' and words[1][0] not in 'aeiou': return 'NP_INVALID_ARTICLE'
  return 'NP'
 if words[0] in {'in','on','at','as','by','of','to'}:return 'PP'
 if words[0] in {'is','are','was','were','emit','emits','spit','tips'}:return 'VP'
 return 'fragment'
def run():
 pairs=json.loads((Path(__file__).parents[1]/'data/mirror_pairs.json').read_text())
 graph=[]
 for p in pairs:
  l=' '.join(p['left']);r=' '.join(p['right'])
  if len(p['left'])==2 and len(p['right'])==2 and len(set(p['left']+p['right']))>2 and role(l)=='NP' and role(r)=='NP':
   graph.append((l,r))
 # Fresh grammatical shells compose graph edges into ordinary scenes. The
 # index supplies vocabulary candidates only; no catalogue sentence is used.
 left_shell=('the poet records {x} beside the river','a careful teacher hears {x} near the window','our young sailor carries {x} home')
 right_shell=('the patient nurse answers {y} before dusk','a quiet neighbor remembers {y} after rain','the old gardener explains {y} at dawn')
 rows=[]
 for l,r in graph[:40]:
  for ls in left_shell:
   for rs in right_shell:
    text=ls.format(x=l)+'; '+rs.format(y=r)+'.'; a=audit(text)
    rows.append({'rendered':text,'graph_edge':{'left':l,'right':r},'audit':a,'provenance':{'graph':'data/mirror_pairs.json search index only','left':'fresh authored grammatical shell','right':'fresh independently authored grammatical shell','catalogue_sentence_borrowed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'word_order_only':False,'repeated_units':False,'self_palindromic_units':False,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']);ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'semordnilap-phrase-graph-constructor-20260920','method':'semordnilap phrase-pair graph index plus independent forward grammatical shells','stats':{'indexed_pairs':len(pairs),'usable_graph_edges':len(graph),'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':ex,'next_construction':'compose two compatible graph edges inside a typed scene grammar while rejecting lexical overlap and testing reader comprehension','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
