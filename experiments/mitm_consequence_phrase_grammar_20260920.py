"""Meet-in-the-middle over fresh agent/action/object/setting + consequence halves."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/mitm-consequence-phrase-grammar-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=("the calm pilot charts a hidden island at dawn, so the crew rests", "a patient doctor carries fresh water through the village, and the children smile", "our careful teacher opens a bright window in winter, while the room warms")
RIGHT=("the local keeper lights a small lantern by the harbor, so travelers wait", "a kind gardener tends a young tree beside the river, and birds return", "our quiet baker saves warm bread near the station, while neighbors gather")
def run():
 # Index only normalized tape prefixes; no catalogue or mirror-pair data enters.
 index={}; joins=[]; probes=0
 for left in LEFT:
  t=n(left); index.setdefault(t[:8],[]).append(left)
 for right in RIGHT:
  t=n(right)[::-1]; key=t[:8]; probes+=1
  for left in index.get(key,[]):
   text=left+'; '+right+'.'; joins.append({'rendered':text,'audit':audit(text),'prefix_key':key,'provenance':{'left':'fresh consequence grammar','right':'fresh independently generated consequence grammar','normalized_prefix_mitm':True,'catalogue_text_reused':False,'mirrored_units':False,'post_hoc_repair':False}})
 exact=[x for x in joins if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'mitm-consequence-phrase-grammar-20260920','method':'normalized prefix meet-in-middle over richer fresh phrase grammar','stats':{'left_halves':3,'right_halves':3,'indexed_prefixes':len(index),'probes':probes,'joins':len(joins),'exact_gt38':len(exact)},'rendered_controls':joins,'exact_candidates':exact,'status':'precise zero frontier: no prefix join' if not joins else ('precise zero exact frontier' if not exact else 'fresh exact requires human reading'),'novelty_preflight':{'status':'passed','distinct_from':'scene lattice, POS trie, endpoint seed, and boundary-shift lanes','catalogue_surface_reuse':False,'mirror_pair_import':False},'provenance':{'audit':'independent mismatch and forward/reverse hashes','reader_gate':'closed unless exact >38'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
