"""Occurrence-based frozen output audit; selection precedes any quality scoring."""
import gzip,hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research/block-seams'
SEED='palindrome-review-20261009-1703-v1'
def norm(text):return re.sub('[^A-Za-z]','',text).lower()
def read(p):return json.loads(gzip.open(p,'rt').read() if p.suffix=='.gz' else p.read_text())
def audit():
 target=OUT/'stratified-output-review-001.json'
 if target.exists():raise RuntimeError('immutable receipt exists')
 rows=[];cells=[];sources=[]
 files=sorted((OUT/'evidence').glob('*results.json*'))+sorted((OUT/'fixtures').glob('*results.json*'))+[OUT/'boundary-root-diagnostic-001-results.json.gz',ROOT/'paper/revision/evidence/results.json']
 for p in files:
  d=read(p);rel=str(p.relative_to(ROOT));sources.append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  paired='results' in d
  cc=d.get('cells',d.get('lanes',d.get('results',[])))
  for i,c in enumerate(cc):
   method=c.get('method',c.get('arm','root_partition'))
   if paired:method+=':penalty='+str(c['penalty'])
   band=c.get('band',[c.get('min_letters',60),c.get('max_letters',119)])
   seed=c.get('seed','heldout');family='7100-7119' if isinstance(seed,int) and 7100<=seed<=7119 else '921-923' if isinstance(seed,int) and 921<=seed<=923 else str(seed)
   cell=rel+'#'+str(c.get('cell_id',c.get('id',c.get('root_number',i))))
   outputs=([c] if c.get('text') else []) if paired else c.get('closures',[])
   cells.append({'id':cell,'method':method,'band':band,'seed':seed,'seed_family':family,'status':c.get('status'),'output_occurrences':len(outputs),'accepted_outputs':int(bool(c.get('strict_admitted_selected_output',False))) if paired else sum(bool(x.get('accepted')) for x in outputs)})
   for j,o in enumerate(outputs):
    text=o.get('text',o.get('raw_assembled_text',''));t=norm(text)
    # Independent two-pointer checker over raw ASCII letters.
    raw=[ch.lower() for ch in text if 'A'<=ch<='Z' or 'a'<=ch<='z'];exact=bool(t) and all(raw[k]==raw[-1-k] for k in range(len(raw)//2))
    assert exact==(bool(t) and t==t[::-1])
    rows.append({'id':cell+'#output-'+str(j),'cell':cell,'source':rel,'method':method,'band':band,'seed':seed,'seed_family':family,'text':text,'exact':exact,'letters':len(t),'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'recorded_grammar_complete':o.get('grammar_complete_as_rendered'),'recorded_accepted':o.get('accepted',c.get('strict_admitted_selected_output')),'source_records':o.get('source_records',[]),'historical_strict_checks':o.get('strict_admission_checks')})
 groups=defaultdict(list)
 for row in rows:groups[(row['source'],row['method'],tuple(row['band']),row['seed_family'])].append(row)
 strata=[];selected=[]
 for key,rr in sorted(groups.items()):
  n=len(rr);k=math.ceil(n/10)
  ranked=sorted(rr,key=lambda x:hashlib.sha256((SEED+'|'+x['id']).encode()).hexdigest())
  chosen=ranked[:k];selected.extend(chosen);strata.append({'key':list(key),'denominator':n,'sample_count':k,'sampling_fraction':k/n,'ids':[r['id'] for r in chosen]})
 controls=read(OUT/'derived-paragraph-controls-001.json')['candidates']
 summary=[]
 for method in sorted({c['method'] for c in cells}):
  cs=[c for c in cells if c['method']==method];rr=[r for r in rows if r['method']==method];unique=len({r['normalized_sha256'] for r in rr})
  summary.append({'method':method,'attempts':len(cs),'attempts_with_output':sum(c['output_occurrences']>0 for c in cs),'attempts_without_output':sum(c['output_occurrences']==0 for c in cs),'output_occurrences':len(rr),'unique_normalized_outputs':unique,'duplicate_occurrence_rate':(len(rr)-unique)/len(rr) if rr else None,'exact_occurrences':sum(r['exact'] for r in rr),'statuses':dict(Counter(c['status'] for c in cs)),'accepted_output_occurrences':sum(c['accepted_outputs'] for c in cs)})
 record={'scope':'Frozen block/seam runs 001-007, all-root diagnostic, and complete 360-attempt paired comparison. Other repository artifacts are discovery-only until schema/provenance verified; no global all-history completeness claim.','plan':{'selection_seed':SEED,'denominator':'All presented closure/output occurrences, before deduplication or quality filtering; failure cells retained separately. Intermediate proposals are not outputs.','strata':'source run, method/penalty, requested length band, seed family','rounding':'ceil(0.10*N) independently per nonempty stratum; empty strata get zero. Overall fraction can exceed 10% due small strata.','ranking':'SHA256(seed|stable occurrence id), ascending','quality':'Individual grammar, readability, cross-sentence coherence, repetition, length and known-example derivation; automated and native-model review separate; no new human scores.','resource_bound':'One local CPU worker; profiles <=10s each, tests <=60s; next controlled ablation at most 120s aggregate, no cloud/GPU/paid calls. Budgets and seeds fixed in preflight; no stale nine-memos seed.','ablations':'Full vs partial components, seam feasibility, composition; caching paired as efficiency only and required output-equivalent under action-limited fixtures. No linguistic conclusions from wall-time-only differences.'},'sources':sources,'method_summary':summary,'cells':cells,'strata':strata,'output_occurrences':len(rows),'sample_occurrences':len(selected),'all_outputs_exact':all(r['exact'] for r in rows),'all_output_checks':[{k:r[k] for k in ('id','exact','letters','normalized_sha256')} for r in rows],'sample':selected,'known_controls':controls,'human_ratings':None,'review_status':'Awaiting independent native readability review; quality scores not yet assigned.'}
 target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'outputs':len(rows),'sample':len(selected),'summary':summary},indent=2))
if __name__=='__main__':audit()
