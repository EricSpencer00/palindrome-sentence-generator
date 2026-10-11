"""Preserved proposal-stage sampling of explicitly verified representative families."""
import hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
SOURCES=[('span_cover','runs/adaptive-crossword-span-cover-20260915.json','rendered_candidates'),('typed_center','runs/typed-overhang-center-search-20260921.json','candidates'),('dependency_chart','runs/dependency-attribute-grammar-chart-20260915.json','rendered_candidates'),('role_trie','runs/role-compatible-character-trie-20260918.json','rendered_candidates'),('scene_lattice','runs/scene-argument-lattice-20260920.json','candidates'),('independent_clause_trie','runs/independent-clause-seam-trie-20260920.json','candidates'),('semantic_frame_tape','runs/semantic-frame-tape-solver-20260916.json','candidates')]
SEED='expanded-representatives-20261009-v1'
def norm(s):return re.sub('[^A-Za-z]','',s).lower()
def run():
 output=OUT/'expanded-representative-sample-001.json'
 if output.exists():raise RuntimeError('immutable output exists')
 sources=[];rows=[];groups=defaultdict(list)
 for method,path,key in SOURCES:
  p=ROOT/path;d=json.loads(p.read_text());items=d[key]
  if method=='span_cover':assert [r for run in d['runs'] for r in run['rows']]==items
  for i,r in enumerate(items):
   text=r.get('text',r.get('rendered',r.get('rendered_text')));assert isinstance(text,str)
   tape=norm(text);raw=[c.lower() for c in text if c.isascii() and c.isalpha()];exact=bool(tape) and tape==tape[::-1];assert exact==(bool(raw) and all(raw[j]==raw[-j-1] for j in range(len(raw)//2)))
   band='0-29' if len(tape)<30 else '30-59' if len(tape)<60 else '60-119' if len(tape)<120 else '120+'
   family=str(r.get('seed',r.get('goal','unrecorded deterministic family')))
   row={'id':method+':'+str(i),'approach':method,'source':path,'source_index':i,'stage':'persisted rendered proposals/tapes, before exactness and linguistic filtering','text':text,'letters':len(tape),'band':band,'seed_family':family,'exact':exact,'normalized_sha256':hashlib.sha256(tape.encode()).hexdigest(),'historical_audit':r.get('audit',r.get('failed_checks',r.get('readability_diagnostic'))),'historical_provenance':r.get('provenance',d.get('provenance')),'human_score':None}
   rows.append(row);groups[(method,band,family)].append(row)
  cs=[r for r in rows if r['approach']==method];unique=len({r['normalized_sha256'] for r in cs})
  sources.append({'approach':method,'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'schema_key':key,'persisted_proposal_denominator':len(cs),'exact':sum(r['exact'] for r in cs),'nonexact_failure_count':sum(not r['exact'] for r in cs),'unique_normalized':unique,'duplicate_occurrence_rate':1-unique/len(cs),'original_stats':d.get('stats'),'original_run_units':len(d['runs']) if isinstance(d.get('runs'),list) else None,'original_candidate_count':d.get('candidate_count',d.get('candidate_rows')),'output_exhaustiveness':'All records in the declared persisted proposal list, not necessarily every attempted/internal search state.','copy_handling':'Top-level list used once; span-cover nested run copies verified identical and excluded; artifacts/runs copies excluded by canonical source choice.'})
 manifest=[];samples=[]
 for key,rr in sorted(groups.items()):
  k=math.ceil(len(rr)/10);chosen=sorted(rr,key=lambda r:hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest())[:k]
  manifest.append({'approach':key[0],'length_band':key[1],'seed_family':key[2],'denominator':len(rr),'sample_count':k,'ids':[r['id'] for r in chosen]});samples.extend(chosen)
 # Preserve failed zero-output family separately, never invent a quality score.
 empty=json.loads((ROOT/'experiments/sentence_intersection-results.json').read_text())
 record={'coverage':'Seven representative additional mechanisms with direct text and provenance schemas verified; full persisted proposal lists audited. Original frozen sample is unchanged. This is representative family coverage, not all 2337 identifier families.','selection_seed':SEED,'rounding':'ceil(10%) per approach, persisted length band and recorded seed/goal family; zero strata sample zero; unrecorded seeds not guessed','sources':sources,'zero_exact_family':{'approach':'sentence_intersection','source':'experiments/sentence_intersection-results.json','proposed':empty['proposed'],'exact_pairs':len(empty['pairs']),'exact_centres':len(empty['centres']),'quality_sample':0,'examples_excluded':True},'proposal_occurrences':len(rows),'all_exactness_checks':[{'id':r['id'],'exact':r['exact'],'letters':r['letters'],'sha256':r['normalized_sha256']} for r in rows],'strata':manifest,'sample':samples,'quality_status':'Awaiting independent review; historical audits are evidence, not fresh human ratings. Proposal-conditioned quality must remain separate from exact-survivor quality and attempt-level generation success.','unresolved_denominators':'Missing attempt seeds/statuses or capped/logged subsets remain explicit in source schema; no invention of total experiment success rates.'}
 output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'proposal_occurrences':len(rows),'sample_n':len(samples),'sources':sources},indent=2)[:7000])
if __name__=='__main__':run()
