import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/per-value-domain-update-before-assignment-20260917.json'
def test_domain_updates_and_audit():
 d=json.loads(RUN.read_text());assert d['candidate_count']==1
 r=d['candidates'][0];assert len(r['domain_updates'])==6 and r['support_by_value']
 assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert d['candidates'][0]['anti_shortcut']['intact_prose'] and d['candidates'][0]['novelty_preflight']['signature']
