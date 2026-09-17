import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).parents[1]/'experiments/affix_residual_transducer_20260917.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def test_live_transducer_has_intact_diagnostics():
 r=m.run(); assert r['config']['independent_affix_transducers']; assert r['config']['live_residual_obligations']; assert r['stats']['rendered']
 assert all(x['independent_exact_audit']['exact'] for x in r['closures'])
 assert all('provenance' in x and 'text' in x for x in r['diagnostic_witnesses'])
def test_not_posthoc_reverse():
 r=m.run(); assert not r['config']['fixed_tape']; assert not r['config']['posthoc_reverse']; assert r['config']['productive_tense_and_number']; assert r['config']['auxiliary_frames']; assert r['config']['state_level_no_repeat']; assert r['novelty_preflight']['excluded']['rlaif']
