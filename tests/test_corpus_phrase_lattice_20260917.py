import json
from pathlib import Path

def test_corpus_phrase_lattice_is_independent_and_long():
    p = Path(__file__).parents[1] / "runs" / "corpus-phrase-lattice-20260917.json"
    d = json.loads(p.read_text())
    assert d["construction"]["max_depth"] == 5
    assert d["construction"]["states"] > 0
    assert min(x["letters"] for x in d["candidates"]) >= 100
    assert all(x["rendered"] and "independent_two_pointer" in x for x in d["candidates"])
    assert d["provenance"]["validation"].startswith("independent")
