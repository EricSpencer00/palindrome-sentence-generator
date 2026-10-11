from experiments.expanded_phrase_coverage_015 import frames
from llm_palindrome.bidirectional_lexical import GrammarDAG,exact_grammar_palindromes

def test_extended_grammar_exact_results_and_real_connectors():
 fs=frames();g=GrammarDAG(fs,1);paths,receipt=exact_grammar_palindromes(g,max_work=2000000,seconds=3,max_paths=5000)
 rows=[g.materialize(p) for p in paths];assert receipt['complete'];assert len(rows)==6
 assert all(r['tape']==r['tape'][::-1] for r in rows)
 assert any(r['text']=='Was it a rat I saw.' for r in rows)
 assert any(any(s.role=='causal_connector' for s in f.slots) for f in fs)
 assert not any('causal' in r['sentences'][0]['frame'] for r in rows)
 g.closure.cache_clear();g.transitions.cache_clear()
