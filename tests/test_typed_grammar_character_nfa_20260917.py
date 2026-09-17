from experiments.typed_grammar_character_nfa_20260917 import run, oracle_rejects_one_character_fake
def test_typed_nfa_no_cartesian_products():
 x=run();assert x['novelty_preflight']['cartesian_sentence_products_materialized'] is False;assert x['expanded_states']>1;assert oracle_rejects_one_character_fake()
 assert x['readability_applied_after_exact_path'] is True;assert x['expanded_states']<=500
 assert all(p['audit']['exact'] for p in x['paired_paths'])
