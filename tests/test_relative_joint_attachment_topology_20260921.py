import json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; M=runpy.run_path(str(ROOT/'experiments/relative_joint_attachment_topology_20260921.py'))
def test_joint_attachment_search():
    M['main'](); d=json.loads((ROOT/'runs/relative-joint-attachment-topology-20260921.json').read_text())
    assert d['internal_pairs']==d['candidate_count']>0
    assert d['exact_count']==0 and d['reader_eligible'] is False
    assert d['stats']['longest_letters']>38
    for r in d['rendered_candidates']:
        assert r['internal_fill']['internally_compatible'] is True
        assert r['audit']['forward_sha256']!=r['audit']['reverse_sha256']
        assert r['audit']['letters']>38
def test_nonpalindrome_audit():
    assert M['audit']('The patient reader.')['two_pointer_exact'] is False
