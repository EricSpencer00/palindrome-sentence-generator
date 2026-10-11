import pytest
from llm_palindrome.typed_phrase_compiler import PhraseRule,compile_phrase_rules,repetition_penalty
from llm_palindrome.bilateral_seams import AllChunkCentersGrammar,OwedSideBilateralGrammar

def test_unlicensed_interface_and_duplicate_feature_rejected():
 r=PhraseRule('START','END','Do not nod.','imperative',(),'test')
 with pytest.raises(ValueError,match='unlicensed'):compile_phrase_rules('p',[r],set())
 r=PhraseRule('START','END','Do not nod.','imperative',(('actor','a'),('actor','b')),'test')
 with pytest.raises(ValueError,match='duplicate feature'):compile_phrase_rules('p',[r],{('START','END')})

def test_shared_interfaces_cannot_bypass_referent_conflict():
 rules=[PhraseRule('START','MID','Draw a ward. ','command',(('actor','reader'),),'donor'),PhraseRule('MID','END','I draw a ward.','reply',(('actor','other'),),'donor')]
 for cls in (AllChunkCentersGrammar,OwedSideBilateralGrammar):
  result=cls(compile_phrase_rules('p',rules,{('START','MID'),('MID','END')})).search(max_steps=2,seconds=2)
  assert result['receipt']['status']=='complete_bounded_depth'
  assert not result['outputs']

def test_repetition_cost_is_finite_and_not_compiler_veto():
 rules=[PhraseRule('START','MID','Draw a ward. ','command',(),'donor'),PhraseRule('MID','END','I draw a ward.','reply',(),'donor')]
 result=OwedSideBilateralGrammar(compile_phrase_rules('p',rules,{('START','MID'),('MID','END')})).search(max_steps=2,seconds=2)
 assert result['outputs']
 assert 0<repetition_penalty(result['outputs'][0]['text'])<100
