import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.grammar_product_certificate_20260917 import run


def test_product_certificate_is_bounded_and_exact():
    report = run()
    assert report["word_boundary_states_retained"]
    assert report["bruteforce_crosscheck"] == {"bound": 12, "exact": True, "matched": len(report["shortest_witnesses"]), "method": "enumerate accepted paths then compare each character with its reverse"}
    assert report["productive_scc_certificate"]["productive"] is False
    assert report["reader_success"] is False
    assert report["provenance"]["catalogue_imported"] is False
    assert report["provenance"]["posthoc_reverse"] is False
    assert Path("artifacts/grammar_product_certificate_20260917.json").exists()
