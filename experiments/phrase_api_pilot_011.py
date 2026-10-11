"""Immutable headless batch commands; no model or paid API calls.
python3 -m experiments.phrase_api_pilot_011 prepare
python3 -m experiments.phrase_api_pilot_011 proposals /absolute/proposals.jsonl BATCH_ID
python3 -m experiments.phrase_api_pilot_011 ratings /absolute/ratings.jsonl BATCH_ID
"""
import argparse,json,hashlib,re
from pathlib import Path
from collections import Counter
from llm_palindrome.phrase_api_loop import validate_proposal,blind_batch,validate_ratings,scored_rows,METHODS
OUT=Path('research/block-seams/phrase-api-loop-011')
def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def lines(p,rows):p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
def fresh(path):
 if path.exists():raise ValueError('batch already exists; inspect saved outcome instead of overwriting')
 path.mkdir(parents=True)
def key(s):
 if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',s):raise ValueError('invalid batch ID')
 return s
def prepare():
 dest=OUT/'baseline-review';fresh(dest);baseline=json.loads((OUT/'baseline-provenance.json').read_text());rows,mapping=blind_batch(baseline,[])
 lines(dest/'blind.jsonl',rows);write(dest/'private-map.json',mapping);write(dest/'receipt.json',dict(status='prepared_unrated',texts=len(rows),paid_api_calls=0,new_generation=0,durably_saved_to_library=False))
 return dest

def proposals(src,bid):
 data=Path(src).read_bytes();raw=jsonl(src);dest=OUT/'proposal-batches'/key(bid);fresh(dest);(dest/'raw.jsonl').write_bytes(data)
 ids=[p.get('proposal_id') for p in raw];counts=Counter(p.get('method') for p in raw)
 bound_errors=[]
 if len(raw)>24:bound_errors.append('total_proposal_budget_exceeded')
 if any(counts[m]>8 for m in METHODS):bound_errors.append('method_proposal_budget_exceeded')
 if len(ids)!=len(set(ids)):bound_errors.append('duplicate_proposal_ids')
 admitted=[validate_proposal(p) for p in raw];lines(dest/'validated.jsonl',admitted)
 write(dest/'receipt.json',dict(status='rejected_batch' if bound_errors else 'validated_unrated',source_sha256=hashlib.sha256(data).hexdigest(),attempts=len(raw),method_counts=dict(counts),accepted=sum(r['mechanically_admitted'] for r in admitted),failed=sum(not r['mechanically_admitted'] for r in admitted),batch_errors=bound_errors,model_quality_results=0,human_approved=0,durably_saved_to_library=False))
 if bound_errors:return dest
 baseline=json.loads((OUT/'baseline-provenance.json').read_text());rows,mapping=blind_batch(baseline,admitted,seed='phrase-api-loop-011|'+bid)
 lines(dest/'blind-comparison.jsonl',rows);write(dest/'private-map.json',mapping);return dest

def ratings(src,bid):
 bid=key(bid);generation=OUT/'proposal-batches'/bid
 if not (generation/'private-map.json').exists():raise ValueError('no valid prepared comparison batch')
 dest=OUT/'rating-batches'/bid;fresh(dest);data=Path(src).read_bytes();(dest/'raw.jsonl').write_bytes(data);raw=jsonl(src);mapping=json.loads((generation/'private-map.json').read_text())
 try:validate_ratings(raw,mapping)
 except ValueError as exc:write(dest/'receipt.json',dict(status='rejected_ratings',reason=str(exc),missing_scores_inferred=0,source_sha256=hashlib.sha256(data).hexdigest(),durably_saved_to_library=False));return dest
 scored=scored_rows(raw,mapping);lines(dest/'scored.jsonl',scored)
 groups={}
 props={r['proposal_id']:r for r in jsonl(generation/'validated.jsonl')}
 for arm in ('baseline',)+METHODS:
  relevant=[r for r in scored if any(o['source']=='baseline' if arm=='baseline' else o['source']=='proposal' and props[o['source_id']]['method']==arm for o in r['occurrences'])]
  groups[arm]=dict(unique_texts=len(relevant),means={k:sum(r[k] for r in relevant)/len(relevant) if relevant else None for k in ['grammar','readability','coherence','repetition_burden','letters']},padding_or_loop=sum(r['padding_or_loop'] for r in relevant),readable_and_coherent=sum(r['readability']>=3 and r['coherence']>=3 for r in relevant))
 write(dest/'comparison.json',dict(groups=groups,interpretation='small fixed model pilot; no human or broad causal claim; lost historical detailed scores not reused'))
 write(dest/'receipt.json',dict(status='complete_validated_model_pilot',source_sha256=hashlib.sha256(data).hexdigest(),expected_unique_texts=len(mapping),scored_unique_texts=len(raw),missing=0,duplicate=0,unknown=0,missing_scores_inferred=0,durably_saved_to_library=False));return dest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','proposals','ratings']);p.add_argument('source',nargs='?');p.add_argument('batch_id',nargs='?');a=p.parse_args()
 if a.mode=='prepare':print(prepare())
 else:print(globals()[a.mode](a.source,a.batch_id))
