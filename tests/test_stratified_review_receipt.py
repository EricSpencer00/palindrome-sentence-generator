import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_selection_is_stratified_and_precedes_quality():
 d=json.loads((ROOT/'research/block-seams/stratified-output-review-001.json').read_text())
 ids={r['id'] for r in d['sample']}
 assert len(ids)==42 and sum(s['sample_count'] for s in d['strata'])==42
 assert sum(s['denominator'] for s in d['strata'])==318
 assert all(s['sample_count']==math.ceil(s['denominator']/10) for s in d['strata'])
 assert all(r['exact'] for r in d['all_output_checks'])
 assert len(d['cells'])==428 and d['human_ratings'] is None
 assert all(hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()==s['sha256'] for s in d['sources'])
 assert sum(c['output_occurrences'] for c in d['cells'])==318
 assert len(ids)==len(d['sample'])
def test_individual_judgments_keep_all_occurrences_and_separate_human_scores():
 p=ROOT/'research/block-seams/stratified-output-review-001.json';sample=json.loads(p.read_text());d=json.loads((p.parent/'stratified-output-judgments-001.json').read_text())
 assert d['sample_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 assert {j['id'] for j in d['judgments']}=={r['id'] for r in sample['sample']}
 assert all(j['human_score'] is None for j in d['judgments'])
