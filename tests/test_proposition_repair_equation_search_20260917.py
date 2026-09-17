import json
from experiments.proposition_repair_equation_search_20260917 import OUT, main, tape

def test_proposition_repair_run_has_independent_audits_and_no_shortcut():
    main()
    report = json.loads(OUT.read_text())
    assert report["novelty_preflight"]["passed"]
    assert report["summary"]["expanded"] == 390625
    assert report["summary"]["distinct_products"] > 0
    assert report["summary"]["reader_eligible"] == 0
    assert report["best_frontier"]["audit"]["independent_agreement"]
    assert report["best_frontier"]["provenance"]["semantic_slot_repair"]
    assert report["best_frontier"]["provenance"]["mechanically_admitted"] is False
    assert tape(report["best_frontier"]["rendered"]) == tape(report["best_frontier"]["rendered"])[::-1] or not report["best_frontier"]["exact"]
