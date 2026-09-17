import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/cross-side-witness-domain-fixed-point-20260917.json'
def test_fixed_point_and_audit():
 d=json.loads(RUN.read_text());r=d['candidates'][0];assert len(r['fixed_point_history'])>1
 assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());r=d['candidates'][0];assert r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature']
