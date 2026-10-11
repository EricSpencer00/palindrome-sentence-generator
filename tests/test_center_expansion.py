import pytest
from llm_palindrome.center_expansion import *

def test_suss_final_s_is_mirrored_not_the_center():
 assert norm(SEED)==L+'sus'+L[::-1]
 assert seed_span(1)['original_center']=='ssuss'
 assert norm('suss')!=norm('suss')[::-1]
 with pytest.raises(ValueError):replace_center(0,'suss')

def test_nested_phrase_changes_both_lexical_seams():
 q=nested_center('tap Pat');r=replace_center(1,q);p=SeedSeamParser().parse(r['tape'])
 assert [x['text'] for x in p]==["Some men, if editors tap Pat's roti, define memos."]
 assert r['letters']==38 and p[0]['bindings']['owner']=='pat1'
 assert len(p[0]['seam_roles'])==2

def test_exact_insertion_can_still_fail_agreement_or_arguments():
 parser=SeedSeamParser()
 for phrase in ["Don't nod",'reward a drawer','step on no pets']:
  r=replace_center(0,phrase);assert r['tape']==r['tape'][::-1]
  assert not parser.parse(r['tape'])

def test_right_roti_resegmentation_has_full_clause_derivation():
 r=replace_center(0,'draw a ward');p=SeedSeamParser().parse(r['tape'])
 assert p[0]['text']=='Some men, if editors draw awards, rot; I define memos.'
 assert 'NP(I)' in p[0]['derivation']
