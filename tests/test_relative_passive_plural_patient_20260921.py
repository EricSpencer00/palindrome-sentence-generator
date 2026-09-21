import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=runpy.run_path(str(R/'experiments/relative_passive_plural_patient_20260921.py'))
def test_plural_passive_patient():
 M['main']();d=json.loads((R/'runs/relative-passive-plural-patient-20260921.json').read_text());assert d['candidate_count']==2 and d['exact_count']==0 and d['novelty_preflight']['all_rendered_new'] and d['stats']['longest_letters']>38
 for x in d['rendered_candidates']:assert x['semantic_state']['agreement_propagated'] and x['audit']['forward_sha256']!=x['audit']['reverse_sha256']
