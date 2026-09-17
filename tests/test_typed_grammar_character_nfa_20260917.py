from experiments.typed_grammar_character_nfa_20260917 import run
def test_typed_nfa_no_cartesian_products():
 x=run();assert x['novelty_preflight']['cartesian_sentence_products_materialized'] is False
 assert x['readability_applied_after_exact_path'] is True;assert x['expanded_states']<=500
 assert all(p['audit']['exact'] for p in x['paired_paths'])
