import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/resolved-position-mask-grammar-dp-20260917.json'
def test_masks_and_audits_exist():
 d=json.loads(RUN.read_text());assert d['candidate_count']==len(d['candidates'])>0
 for r in d['candidates']:
  assert r['mask_width']==len(r['resolved_position_mask'])>0
  assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_layers_and_provenance():
 d=json.loads(RUN.read_text());assert len(d['layers'])==len(d['candidates'][0]['slots'])
 assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
