import json,hashlib,math
from pathlib import Path
from llm_palindrome.admission import normalize_letters
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'research/block-seams'
def test_feedback_and_source_provenance_are_preserved():
 h=json.loads((B/'human-five-output-feedback-001.json').read_text())
 assert len(h['whole_output_labels'])==5 and h['fragment_approvals']==[]
 assert all(r['human_label']=='BAD' and r['numeric_human_rating'] is None for r in h['whole_output_labels'])
 d=json.loads((B/'fragment-salvage-recomposition-001.json').read_text());assert d['plan']['source_bank_sha256']==hashlib.sha256((ROOT/'data/v3_bank.json').read_bytes()).hexdigest()
 assert all(d['plan']['v3_components'].values())
def test_complete_salvage_is_exact_and_semantic_filter_preserves_gates():
 d=json.loads((B/'fragment-salvage-recomposition-001.json').read_text());a,c,v=d['results']
 assert [r['status'] for r in d['results']]==['completed']*3
 assert [r['exact_output_occurrences'] for r in d['results']]==[528,67,82]
 for result in d['results']:
  assert all(normalize_letters(r['text'])==normalize_letters(r['text'])[::-1] for r in result['outputs'])
  assert all(s['sample_count']==math.ceil(s['denominator']/10) for s in result['sampling_manifest'])
 assert all(not r['inanimate_sees_clauses'] for result in (c,v) for r in result['outputs'])
 assert {r['text'] for r in c['outputs'] if r['mechanically_eligible']}=={r['text'] for r in v['outputs'] if r['mechanically_eligible']}
 assert d['human_review_queue']==[]
