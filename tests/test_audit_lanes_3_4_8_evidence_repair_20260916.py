import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'experiments/audit_lanes_3_4_8_evidence_repair_20260916.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def test_targeted_audit_repair():
 audit.main();d=json.loads((ROOT/'runs'/(audit.ID+'.json')).read_text())
 assert d['novelty_preflight']['audit_only'] and d['novelty_preflight']['search_performed'] is False
 assert d['lanes']['3_dependency_seam']['reverse_sha256']
 assert d['lanes']['8_inflection_clitic']['anti_shortcut']['repeated_canonical_unit'] is True
 assert d['lanes']['4_morphology_transducer']['all_independently_audited']
