"""Finite center-block insertion plus independently bounded longer lexical probe.
Semantic features are authored diagnostics, never independent quality scores.
"""
import gzip,hashlib,json,math,time
from collections import Counter,defaultdict
from pathlib import Path
from dataclasses import replace
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.bidirectional_lexical import GrammarDAG,exact_grammar_palindromes,SearchBudgetExceeded
from experiments.structural_breadth_palindromes_20261009 import breadth_frames
from experiments.bidirectional_lexical_completion_20261009 import output_row
OUT=Path('research/block-seams/semantic-assembly-010');SEED='semantic-assembly-010'
FROZEN=Path('research/block-seams/structural-breadth-009')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(row):return hashlib.sha256((SEED+'|'+row['id']).encode()).hexdigest()
def event(s):
 d=dict(zip(s['roles'],s['words']));f=s['frame']
 if 'delivery' in f:return ('deliver',d.get('addressee',d.get('agent',d.get('human_agent'))),d.get('recipient',d.get('human_theme')), 'request' if 'request' in f else 'report')
 if 'revile' in f or 'criticism' in f:return ('criticize',d.get('human_agent',d.get('human_plural_agent')),d.get('human_theme'),'past')
 if f=='stopping_request':return ('stop',None,d.get('human_theme'),'request')
 if 'reference' in f:return ('refer',d.get('addressee',d.get('agent')),d.get('human_theme'),'request')
 return ('other',None,None,None)
def features(row):
 es=[event(s) for s in row['sentences']];links=[]
 # Directed roles/actions, not undirected shared-noun adjacency. Still does
 # not assert causal truth, intention, temporal ordering or human coherence.
 for i,(a,b) in enumerate(zip(es,es[1:])):
  if a[0]=='deliver' and b[0]=='criticize' and a[2] and a[2]==b[1] and 'stressed' in row['sentences'][i+1]['words']:links.append([i,'delivery_recipient_then_affected_critic'])
  if a[0]=='criticize' and b[0]=='stop' and a[1] and a[1]==b[2]:links.append([i,'critic_then_stop_same_actor'])
  if a[0]=='criticize' and b[0]=='refer' and a[2] and a[2]==b[2]:links.append([i,'criticized_person_then_referral'])
 agents={e[1] for e in es if e[1]};verbs={e[0] for e in es if e[0]!='other'}
 return dict(role_transition_links=links,link_count=len(links),link_fraction=len(links)/max(1,len(es)-1),distinct_agents=len(agents),action_diversity=len(verbs),unmodeled_events=sum(e[0]=='other' for e in es),repeated_sentences=row['repeated_sentences'])
def rank(row):
 f=features(row);return (-f['link_fraction'],-f['link_count'],f['unmodeled_events'],f['distinct_agents'],row['repeated_sentences'],stable(row))
def boundary(row):
 n=0
 for i,s in enumerate(row['sentences'][:-1]):
  n+=len(norm(s['text']))
  if n*2==row['letters']:return i+1
 return None
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def write_rows(name,rows):
 with gzip.open(OUT/name,'wt') as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
def run():
 OUT.mkdir(parents=True,exist_ok=True);t0=time.monotonic()
 frozen={str(p):digest(p) for p in FROZEN.rglob('*') if p.is_file()};save('frozen-inputs.json',frozen)
 plan=dict(seed=SEED,center_outer_count=200,center_inner_count=100,center_proposals=20000,selection='SHA256 order independent of semantic rank; <=4 clause frozen009 blocks',lexical_clauses=[5,6],lexical_max_work_per_clause=2000000,lexical_max_seconds_per_clause=12,lexical_max_paths_per_clause=20000,models=0,cpu_only=True,comparison='within identical length/repeated-sentence strata, top semantic diagnostic decile vs SHA256 random decile; same frozen and new populations; selection features are not independent evaluation',sample='ceil10% per method x clause count x repeated-sentence count; retain duplicate proposals and failures',stop='20k insertion proposals and two lexical probes, no adaptive follow-up')
 save('plan.json',plan)
 save('parent-review-summary.json',dict(source='parent message; raw scores cloud only, no local per-ID integration',unique_texts=2686,fresh_ratings=2273,reused_exact_ratings=413,chunks=42,means=dict(grammar=3.77,readability=3.20,coherence=1.86,variety=2.35),shares_ge3=dict(grammar=.970,readability=.890,coherence=.040,variety=.504),grammar_ge3_coherence_le2_share=.931,zip_sha256='fb27031a86aafb709608b68c45e900d75df98f79650de83d2f005fd8151ea03e',human_ratings=False))
 old=[json.loads(x) for x in gzip.open(FROZEN/'all-deduplicated-outputs.jsonl.gz','rt')];oldset={r['tape'] for r in old}
 outer=sorted((r for r in old if boundary(r) is not None),key=stable)[:200];inner=sorted(old,key=stable)[:100]
 save('selected-blocks.json',dict(outer_ids=[r['id'] for r in outer],inner_ids=[r['id'] for r in inner]))
 rows=[];fail=[]
 for a in outer:
  k=boundary(a)
  for b in inner:
   ss=a['sentences'][:k]+b['sentences']+a['sentences'][k:];text=' '.join(s['text'] for s in ss)
   if norm(text)!=norm(text)[::-1]:fail.append(dict(outer=a['id'],inner=b['id'],reason='exactness'));continue
   r=output_row('center_insert',dict(text=text,sentences=ss),dict(outer_id=a['id'],inner_id=b['id'],center_boundary=k));r['new_to009']=r['tape'] not in oldset;r['semantic_diagnostic']=features(r);rows.append(r)
 write_rows('insertion-raw.jsonl.gz',rows);save('insertion-failures.json',fail)
 # Same vocabulary and licensed roles; restrict humans to two recurrent actors.
 names={'Noel','Anna'};humanroles={'addressee','human_theme','human_agent','recipient'}
 allowed={'food_delivery_request','stressed_criticism','past_revile','reference_request','stopping_request','person_delivery_request'}
 fs=[]
 for f in breadth_frames():
  if f.name not in allowed:continue
  slots=[]
  for s in f.slots:
   words=tuple(w for w in s.words if s.role not in humanroles or w in names)
   slots.append(replace(s,words=words))
  if all(s.words for s in slots):fs.append(replace(f,slots=tuple(slots)))
 from dataclasses import asdict
 save('lexical-frames.json',[asdict(f) for f in fs]);receipts=[]
 for n in [5,6]:
  g=GrammarDAG(fs,n)
  with gzip.open(OUT/f'lexical-states-{n}.jsonl.gz','wt') as trace:
   try:paths,receipt=exact_grammar_palindromes(g,max_work=2000000,max_paths=20000,seconds=12,trace=lambda x:trace.write(json.dumps(x)+'\n'))
   except SearchBudgetExceeded as exc:paths=[];receipt=exc.receipt
  receipt['clauses']=n;receipts.append(receipt);local=[]
  for path in paths:
   r=output_row('lexical_long',g.materialize(path),dict(character_arc_ids=path,clauses=n));r['new_to009']=r['tape'] not in oldset;r['semantic_diagnostic']=features(r);local.append(r)
  write_rows(f'lexical-outputs-{n}.jsonl.gz',local);rows+=local;g.closure.cache_clear();g.transitions.cache_clear()
 save('lexical-receipts.json',receipts)
 # All output tokens independently licensed by original009 grammar, which
 # allows arbitrary sentence count; sentence metadata keeps lexical provenance.
 specs=breadth_frames()
 for r in rows:
  assert norm(r['text'])==r['tape']==r['tape'][::-1]
  for s in r['sentences']:
   assert any(f.name==s['frame'] and len(f.slots)==len(s['words']) and all(w in sl.words for w,sl in zip(s['words'],f.slots)) and f.render(s['words'])==s['text'] for f in specs)
 unique={r['tape']:r for r in rows};write_rows('unique-outputs.jsonl.gz',unique.values())
 comparisons=[];sample=[];manifest=[]
 for method,pop in [('frozen009',old),('center_insert',[r for r in rows if r['id'].startswith('center_insert')]),('lexical_long',[r for r in rows if r['id'].startswith('lexical_long')])]:
  strata=defaultdict(list)
  for r in pop:strata[(r['clause_count'],r['letters'],r['repeated_sentences'])].append(r)
  for st,p in sorted(strata.items()):
   k=math.ceil(len(p)/10);control=sorted(p,key=stable)[:k];selected=sorted(p,key=rank)[:k]
   comparisons.append(dict(method=method,stratum=st,population=len(p),k=k,random_ids=[r['id'] for r in control],selected_ids=[r['id'] for r in selected],random_mean_links=sum(features(r)['link_fraction'] for r in control)/k,selected_mean_links=sum(features(r)['link_fraction'] for r in selected)/k))
  # Sampling uses broader declared method/clause/repetition strata, explicit rounding.
  strata=defaultdict(list)
  for r in pop:strata[(r['clause_count'],r['repeated_sentences'])].append(r)
  for st,p in sorted(strata.items()):
   k=math.ceil(len(p)/10);chosen=sorted(enumerate(p),key=lambda z:hashlib.sha256((SEED+'|sample|'+method+'|'+str(z[0])+'|'+z[1]['id']).encode()).hexdigest())[:k]
   manifest.append(dict(method=method,stratum=st,population=len(p),k=k));sample += [dict(method=method,occurrence_index=i,**r) for i,r in chosen]
 save('controlled-comparison.json',comparisons);save('sample-manifest.json',manifest);write_rows('sample.jsonl.gz',sample)
 # Ranked list retained entirely; diversity queue masks names, no human labels.
 ranked=sorted(unique.values(),key=rank);write_rows('ranked-complete-pool.jsonl.gz',ranked)
 queue=[];patterns=set()
 for r in ranked:
  pattern=' '.join('<person>' if w in {'Noel','Leon','Anna','Eve','Liam','Iris'} else w for s in r['sentences'] for w in s['words'])
  if pattern in patterns or r['letters']<=92:continue
  patterns.add(pattern);queue.append(r)
  if len(queue)==5:break
 save('prospective-queue.json',queue)
 assert all(digest(Path(p))==h for p,h in frozen.items())
 summary=dict(proposals_insertion=20000,accepted_insertion=20000-len(fail),failed_insertion=len(fail),raw_output_occurrences=len(rows),unique_outputs=len(unique),duplicate_occurrences=len(rows)-len(unique),new_unique_to009=sum(r['new_to009'] for r in unique.values()),unique_beyond92=sum(r['letters']>92 for r in unique.values()),max_letters=max(r['letters'] for r in rows),max_clauses=max(r['clause_count'] for r in rows),lexical_receipts=receipts,sample_occurrences=len(sample),sample_denominator=len(old)+len(rows),sample_percent=100*len(sample)/(len(old)+len(rows)),verified_output_occurrences=len(rows),frozen_files_unchanged=len(frozen),elapsed_seconds=time.monotonic()-t0,independent_new_quality_reviews=0,human_approved_new_outputs=0,limitation='role transition feature is optimized by construction; selection comparison is diagnostic and cannot establish human or model coherence improvement. Nesting exact paragraphs yields length, not automatically progression.')
 save('summary.json',summary);save('checkpoint.json',dict(status='complete',objective='semantic assembly beyond92letters',next='inspect strongest queue and diagnostic gains before any new independent review',summary=summary));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
