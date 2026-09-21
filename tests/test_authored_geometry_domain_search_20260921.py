import importlib.util
from pathlib import Path

P=Path(__file__).parents[1]/"experiments/authored_geometry_domain_search_20260921.py"
s=importlib.util.spec_from_file_location("ag",P); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

def test_authored_geometry_bank_is_nonempty_and_long():
    rows=m.load()
    assert rows and max(map(lambda x:sum(map(len,x)),rows))>38

def test_boundary_filter_and_audit_are_independent():
    rows=m.load()
    for words in rows:
        assert not m.reflected(tuple(map(len,words)))
        a=m.audit(m.render(words)); assert a["sha256_forward"] != a["sha256_reverse"] or a["pointer_exact"]

def test_run_records_controls_and_reader_gate():
    out=m.run()
    assert out["summary"]["patterns"] >= 1
    assert out["reader_gate"].startswith("closed")
    assert all("source_control" in c and "provenance" in c for c in out["cases"])
