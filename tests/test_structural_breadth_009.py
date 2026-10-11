import itertools
from collections import Counter
from experiments.structural_breadth_palindromes_20261009 import breadth_frames,restricted_graph,GROUPS,vocabulary
from experiments.bidirectional_lexical_completion_20261009 import enumerate_sentences
from llm_palindrome.bidirectional_lexical import exact_grammar_palindromes
from llm_palindrome.admission import normalize_letters as norm

def test_partitioned_product_matches_exhaustive_and_retains_all_derivations():
    specs=tuple(f for f in breadth_frames() if f.name in ('drawing_request','material_delivery_request','no_rider_criticism'))
    partitions=[{'drawing_request'},{'material_delivery_request','no_rider_criticism'}]
    for n in (1,2,3):
        got=[]
        for first in partitions:
            g=restricted_graph(specs,n,first);paths,receipt=exact_grammar_palindromes(g)
            assert receipt['complete']
            for p in paths:
                item=g.materialize(p);assert item['sentences'][0]['frame'] in first;got.append(item['text'])
        bank=enumerate_sentences(specs);expected=[]
        for ss in itertools.product(bank,repeat=n):
            text=' '.join(s['text'] for s in ss);t=norm(text)
            if t==t[::-1]:expected.append(text)
        assert Counter(got)==Counter(expected)

def test_role_breadth_negative_linked_frames_and_proper_relative_punctuation():
    fs=breadth_frames();assert set.union(*GROUPS.values())=={f.name for f in fs}
    v=vocabulary(fs);assert any(x['word']=='rats' and x['source']['lemma']=='rat' for x in v)
    r=next(f for f in fs if f.name=='relative_delivery_request')
    assert r.render(['Noel','deliver','desserts','to','Anna','who','reviled','Leon'])=='Noel, deliver desserts to Anna, who reviled Leon.'
    causal=next(f for f in fs if f.name=='causal_delivery_report')
    assert causal.render(['Sir','I','deliver','desserts','because','Anna','reviled','Leon'])=='Sir, I deliver desserts because Anna reviled Leon.'

def test_new_material_plural_and_report_seams_are_global_exact():
    fs=breadth_frames()
    for families,target in [
        ({'material_delivery_request','no_rider_criticism'},'Noel, deliver red iron. No rider reviled Leon.'),
        ({'animal_delivery_statement','celebrity_revile'},'Noel delivers rats. Stars reviled Leon.'),
        ({'addressed_food_delivery_report','iris_stressed_criticism'},'Sir, I deliver Anna desserts. Stressed Anna reviled Iris.'),
    ]:
        specs=tuple(f for f in fs if f.name in families);g=restricted_graph(specs,2,families);paths,_=exact_grammar_palindromes(g)
        texts=[g.materialize(p)['text'] for p in paths];assert target in texts
        assert all(norm(t)==norm(t)[::-1] for t in texts)
