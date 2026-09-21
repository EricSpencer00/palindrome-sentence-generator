import json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; M=runpy.run_path(str(ROOT/'experiments/relative_arc_outer_topology_20260921.py'))
def test_outer_topology_preserves_controls_and_audit():
    M['main'](); d=json.loads((ROOT/'runs/relative-arc-outer-topology-20260921.json').read_text())
    assert d['outer_pairs']==d['candidate_count']>0
    assert d['exact_count']==0 and d['reader_eligible'] is False
    assert d['stats']['longest_letters']>38
    for r in d['rendered_candidates']:
        assert r['outer_boundary_obligation']['reversed_match'] is True
        assert r['audit']['forward_sha256']!=r['audit']['reverse_sha256']
        assert r['audit']['letters']>38
def test_audit_rejects_mismatch():
    assert M['audit']('A quiet river.')['two_pointer_exact'] is False
