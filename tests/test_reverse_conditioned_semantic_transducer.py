from experiments.reverse_conditioned_semantic_transducer_20260920 import run
def test_transducer_zero_frontier():
 x=run(); assert x['stats']['lexicon_words']>30; assert x['stats']['online_parses']==0; assert x['stats']['exact_gt38']==0
