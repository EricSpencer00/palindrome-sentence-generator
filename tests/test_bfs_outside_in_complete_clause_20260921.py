import json
from pathlib import Path
def test_bfs_has_rendered_complete_controls_and_audits():
 d=json.loads(Path('runs/bfs-outside-in-complete-clause-20260921.json').read_text())
 assert d['stats']['rendered_controls']>0
 assert d['stats']['max_letters']>38
 for r in d['strongest_complete_controls']:
  assert r['provenance']['complete_clause_grammar'] and 'sha256_forward' in r['audit']
