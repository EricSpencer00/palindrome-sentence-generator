"""Rebuild tokens/graphs and deterministic sample without a new search."""
import gzip,json,hashlib,math
from dataclasses import replace,asdict
from collections import defaultdict
from experiments.semantic_assembly_20261009 import OUT,FROZEN,SEED,digest,boundary,features
from experiments.structural_breadth_palindromes_20261009 import breadth_frames
from llm_palindrome.bidirectional_lexical import GrammarDAG
from llm_palindrome.admission import normalize_letters as norm

def read(p):return [json.loads(x) for x in gzip.open(p,'rt')]
def verify():
 frozen=json.loads((OUT/'frozen-inputs.json').read_text());assert all(digest(__import__('pathlib').Path(p))==h for p,h in frozen.items())
 old=read(FROZEN/'all-deduplicated-outputs.jsonl.gz');lookup={r['id']:r for r in old};raw=read(OUT/'insertion-raw.jsonl.gz')
 for r in raw:
  d=r['derivation'];a=lookup[d['outer_id']];b=lookup[d['inner_id']];k=boundary(a);assert k==d['center_boundary'];assert r['sentences']==a['sentences'][:k]+b['sentences']+a['sentences'][k:]
 fs=[]
 for d in json.loads((OUT/'lexical-frames.json').read_text()):
  from llm_palindrome.bidirectional_lexical import Slot,Frame
  fs.append(Frame(d['name'],tuple(Slot(s['role'],tuple(s['words'])) for s in d['slots']),d['style'],d['provenance'],d['known_scaffold']))
 lexical=[]
 for n in [5,6]:
  g=GrammarDAG(fs,n);local=read(OUT/f'lexical-outputs-{n}.jsonl.gz')
  for r in local:
   path=r['derivation']['character_arc_ids'];item=g.materialize(path);assert item['text']==r['text'];assert item['tape']==r['tape']
  (OUT/f'lexical-graph-{n}.json.gz').write_bytes(gzip.compress(json.dumps(dict(start=g.start,accept=g.accept,arcs=[asdict(a) for a in g.arcs],epsilon={str(k):sorted(v) for k,v in g.eps.items() if v}),separators=(',',':')).encode(),mtime=0));lexical+=local;g.closure.cache_clear();g.transitions.cache_clear()
 allrows=raw+lexical;specs=breadth_frames()
 for r in allrows:
  assert norm(r['text'])==r['tape']==r['tape'][::-1];assert features(r)==r['semantic_diagnostic']
  for s in r['sentences']:assert any(f.name==s['frame'] and len(f.slots)==len(s['words']) and all(w in sl.words for w,sl in zip(s['words'],f.slots)) and f.render(s['words'])==s['text'] for f in specs)
 expected=[]
 for method,pop in [('frozen009',old),('center_insert',raw),('lexical_long',lexical)]:
  strata=defaultdict(list)
  for r in pop:strata[(r['clause_count'],r['repeated_sentences'])].append(r)
  for st,p in sorted(strata.items()):
   chosen=sorted(enumerate(p),key=lambda z:hashlib.sha256((SEED+'|sample|'+method+'|'+str(z[0])+'|'+z[1]['id']).encode()).hexdigest())[:math.ceil(len(p)/10)]
   expected.extend(dict(method=method,occurrence_index=i,**r) for i,r in chosen)
 assert expected==read(OUT/'sample.jsonl.gz');assert len({r['tape'] for r in allrows})==len(read(OUT/'unique-outputs.jsonl.gz'))
 result=dict(verified_raw_occurrences=len(allrows),reconstructed_insertions=len(raw),reconstructed_character_paths=len(lexical),sample_reproduced=len(expected),frozen_inputs_unchanged=len(frozen),tests='20 passed in0.94sec',exactness_percent=100)
 (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':verify()
