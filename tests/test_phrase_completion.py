from llm_palindrome.phrase_completion import PhraseCompleter
from llm_palindrome.bilateral_seams import debt

def control_arguments():
 b={'observer':'geese1','patient':'God1','subject_number':'plural'}
 return {'predicate':'perceive','patient_surface':'God','construction':'question_do','bindings':b,'required_bindings':dict(b)}

def test_right_generator_builds_np_then_verb_without_supplied_vp():
 g=PhraseCompleter();r=g.complete_phrase(debt('Do geese',''),'right','perception_vp',control_arguments())
 assert r['receipt']['status']=='complete_finite_grammar'
 assert len(r['phrases'])==1
 p=r['phrases'][0];assert p['text']=='see God' and p['paid_letters']==6
 assert p['debt']=={'side':'left','letters':'e'}
 assert [s['word'] for s in p['trace']]==['God','see']
 assert [s['paid'] for s in p['trace']]==[3,3]
 assert len(p['lexical_parse'])==2 and p['resource_hashes']

def test_wrong_referent_and_agreement_reject_identical_goal_tape():
 for key,value in [('patient','other1'),('subject_number','singular')]:
  args=control_arguments();args['bindings']={**args['bindings'],key:value}
  r=PhraseCompleter().complete_phrase(debt('Do geese',''),'right','perception_vp',args)
  assert not r['phrases'] and r['failures'][0]['reason']=='semantic_or_agreement_conflict'

def test_local_construction_obligations_and_request_cap():
 g=PhraseCompleter(max_requests=1);args={'construction':'modal_deliberation','bindings':{}}
 r=g.complete_phrase({'side':'none','letters':''},'left','opening',args)
 assert {p['text'] for p in r['phrases']}=={'You should pause','You should wait'}
 assert all(p['local_obligations']['construction']=='modal_deliberation' for p in r['phrases'])
 import pytest
 with pytest.raises(RuntimeError,match='request cap'):g.complete_phrase({'side':'none','letters':''},'left','opening',args)

def test_directional_letter_mismatch_is_preserved():
 r=PhraseCompleter().complete_phrase({'side':'left','letters':'xxxx'},'right','perception_vp',control_arguments())
 assert not r['phrases'] and r['failures'][0]['reason']=='letter_mismatch'
