import json
from pathlib import Path

from experiments.recursive_obligation_clause_growth_20260916 import expand, norm, audit


def test_recursive_repair_expands_frontier_and_keeps_gate_closed():
    base = expand('', repair=False)
    repair = expand('', repair=True)
    assert len(repair) > len(base)
    assert len(base) == 16
    assert len(repair) == 128
    assert all(not audit(row['text'])['exact'] for row in base + repair)
    assert all(row['obligation'] == norm(row['text'])[::-1] for row in repair)


def test_run_artifact_records_no_reader_candidates():
    path = Path(__file__).parents[1] / 'runs/recursive-obligation-clause-growth-20260916.json'
    data = json.loads(path.read_text())
    assert data['reader_eligible'] == []
    assert data['base']['candidates'] == []
    assert data['repair']['candidates'] == []
