import pytest
from dataclasses import replace
from llm_palindrome.phrase_inventory_search import PhraseInventoryDAG,Phrase
from llm_palindrome.bidirectional_lexical import exact_grammar_palindromes
from experiments.phrase_inventory_pilot_012 import fixture

def test_phrase_product_matches_expected_partial_word_control():
 inv,plans=fixture();g=PhraseInventoryDAG(inv,plans);paths,receipt=exact_grammar_palindromes(g,max_work=10000,max_paths=100,seconds=2)
 rs=[g.materialize(p) for p in paths];assert len(rs)==7;assert receipt['complete']
 r=next(r for r in rs if r['plan']=='partial_word_question');assert r['text']=='Was it a car or a cat I saw?';assert [p['text'] for p in r['phrases']]==['Was it a ca','r or a ca','t I saw?']
 assert r['tape']==r['tape'][::-1];assert all('.' not in p['text'] for p in r['phrases'])
 with pytest.raises(AssertionError):g.materialize(paths[0][:-1])
 g.closure.cache_clear();g.transitions.cache_clear()

def test_wrong_grammar_seam_rejected():
 inv,plans=fixture();inv[0]=replace(inv[0],exit='WRONG')
 with pytest.raises(ValueError,match='unlicensed phrase seam'):PhraseInventoryDAG(inv,plans)

def test_binding_conflict_retained_without_quality_claim():
 inv=[Phrase('l','Eve, ','START','M','speaker','fixture',(('actor','Eve'),)),Phrase('m','refer ','M','N','verb','fixture'),Phrase('r','Eve.','N','END','theme','fixture',(('actor','Mom'),))]
 plans=[dict(id='fixture',method='test',states=['START','M','N','END'],slots=[['l'],['m'],['r']])];g=PhraseInventoryDAG(inv,plans);paths,_=exact_grammar_palindromes(g,seconds=2)
 r=g.materialize(paths[0]);assert not r['role_consistent'];assert r['binding_conflicts'];assert r['tape']==r['tape'][::-1]
 g.closure.cache_clear();g.transitions.cache_clear()

def test_input_budget_and_native_json_adapter():
 with pytest.raises(ValueError,match='input budget'):PhraseInventoryDAG([Phrase(str(i),'Eve','START','END','np','fixture') for i in range(49)],[])
 g=PhraseInventoryDAG.from_json_api(dict(fragments=[dict(id='a',text='Eve.',entry='START',exit='END',role='human',source='fixture',bindings={})],plans=[dict(id='a',method='fixture',states=['START','END'],slots=[['a']])]))
 paths,_=exact_grammar_palindromes(g,seconds=2);assert g.materialize(paths[0])['text']=='Eve.'
 g.closure.cache_clear();g.transitions.cache_clear()

def test_conditioned_frontier_reports_useful_debt_and_viable_options():
 from llm_palindrome.phrase_inventory_search import fragment_frontier
 inv,plans=fixture();payload=dict(fragments=[dict(id=p.id,text=p.text,entry=p.entry,exit=p.exit,role=p.role,source=p.source,bindings=dict(p.bindings)) for p in inv],plans=plans)
 p=plans[0];d=fragment_frontier(payload,p['id'],[p['slots'][0][0]],[p['slots'][-1][0]])
 assert d['current']['letter_debt']['inner_required_suffix']=='t';assert d['left_grammar_state']=='Q';assert d['right_grammar_state']=='REL'
 assert any(x['compatible'] for x in d['paired_endpoint_alternatives'])
 left_car=p['slots'][1][0];d=fragment_frontier(payload,p['id'],[p['slots'][0][0],left_car],[p['slots'][-1][0]])
 assert d['current']['letter_debt']['inner_required_suffix']=='racat'
 assert any(x['compatible'] and x['text']=='a cat' for x in d['alternatives']['right'])

def test_endpoint_mismatch_rejects_with_actual_chars():
 from llm_palindrome.phrase_inventory_search import fragment_frontier
 payload=dict(fragments=[dict(id='a',text='I saw ',entry='START',exit='N',role='vp',source='fixture'),dict(id='b',text='Mom.',entry='N',exit='END',role='np',source='fixture')],plans=[dict(id='bad',method='fixture',states=['START','N','END'],slots=[['a'],['b']])])
 d=fragment_frontier(payload,'bad');assert d['immediate_rejection'];pair=d['paired_endpoint_alternatives'][0]
 assert pair['letter_debt']['left_char']=='i';assert pair['letter_debt']['mirrored_right_char']=='m';assert not pair['letter_debt']['completion_possible_by_middle_only']

def test_lattice_and_admission_prevent_unconditioned_letter_failure():
 from llm_palindrome.phrase_inventory_search import conditioned_lattice,letter_admit_alternative
 inv,plans=fixture();payload=dict(fragments=[dict(id=p.id,text=p.text,entry=p.entry,exit=p.exit,role=p.role,source=p.source,bindings=dict(p.bindings)) for p in inv],plans=plans);p=plans[0];l=[p['slots'][0][0],p['slots'][1][0]];r=[p['slots'][-1][0]]
 lat=conditioned_lattice(payload,p['id'],l,r);assert lat['exact_witnesses']==1;assert lat['outputs'][0]['text']=='Was it a car or a cat I saw?'
 bad=dict(id='new',text='a dog',entry='ALT',exit='REL',role='np',source='fixture',bindings={});gate=letter_admit_alternative(payload,p['id'],l,r,'right',bad);assert not gate['admitted'];assert gate['reason']=='letter_mismatch';assert gate['mismatch']['first_mismatch']==4
 good=dict(bad,text='a cat');gate=letter_admit_alternative(payload,p['id'],l,r,'right',good);assert gate['admitted'];assert len(gate['witnesses'])==1
 no=dict(id='connector',text=' and ',entry='NP',exit='ALT',role='conjunction',source='fixture',bindings={});gate=letter_admit_alternative(payload,p['id'],l,r,'left',no);assert not gate['admitted'];assert gate['reason']=='no_exact_closure_in_finite_inventory'
 assert all(f['id'] not in ('new','connector') for f in payload['fragments'])
