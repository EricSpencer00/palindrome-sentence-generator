import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.three_region_agreement_repair_20260917 import run, letters

def test_agreement_repair_is_bounded_and_independently_audited():
    payload = run()
    assert payload["stats"]["rendered"] == 32
    assert payload["stats"]["exact"] == 0
    for row in payload["rendered_candidates"]:
        assert len(row["regions"]) == 3
        assert row["audit"]["letters"] == len(letters(row["rendered"]))
        assert row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"]
        assert not row["provenance"]["catalogue_used"]
