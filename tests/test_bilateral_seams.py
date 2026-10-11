from llm_palindrome.bilateral_seams import Chunk,BilateralGrammar,debt,payoff

def test_both_overhangs_and_cancellation():
 assert debt('Was it ',' I saw?')['letters']=='t'
 assert debt('ab','cba')['side']=='right'
 assert debt('ab','ba')['side']=='none'
 assert debt('aba','')['closure_exact']
 assert not debt('ab','')['closure_exact']
 assert not debt('cat','dog')['compatible']
 p=payoff(debt('abc',''), 'right',2,debt('abc','ba'));assert p['paid']==2 and p['ratio']==1
 assert payoff(debt('abc',''),'left',2,debt('abcde',''))['paid']==0

def fixture():
 return [Chunk('L1','Was it a ','START','CA','question','existing question control'),Chunk('L2','ca','CA','R','partial car','existing car lemma'),Chunk('L3','r or a ','R','CAT','car completion/alternative','existing question control'),Chunk('R3','c','CAT','AT','partial cat','existing cat lemma'),Chunk('R2','at','AT','REL','cat completion','existing cat lemma'),Chunk('R1',' I saw?','REL','END','relative observation','existing question control',(('observer','I'),))]

def test_real_three_three_seams_and_exact_closure():
 g=BilateralGrammar(fixture());s=g.initial()
 for pid,side in [('L1','left'),('R1','right'),('L2','left'),('R2','right'),('L3','left'),('R3','right')]:
  c=next(c for c in g.chunks if c.id==pid);s,error=g.extend(s,c,side);assert error is None
 assert s['left_state']==s['right_state'];assert s['debt']['closure_exact'];assert s['left']+s['right']=='Was it a car or a cat I saw?';assert s['left_ids']==['L1','L2','L3'];assert s['right_ids']==['R3','R2','R1']
 result=g.search(max_steps=6);assert len(result['outputs'])==1;assert result['outputs'][0]['tape']==result['outputs'][0]['tape'][::-1]

def test_cycle_novelty_not_collapsed_and_repetition_soft():
 g=BilateralGrammar([Chunk('loop','a','START','START','formal diagnostic','not English prose'),Chunk('end','a','START','END','formal diagnostic','not English prose')]);r=g.search(max_steps=6,max_outputs=20)
 assert {o['letters'] for o in r['outputs']}==set(range(1,7));assert any(o['repeat_cost']>0 for o in r['outputs']);assert r['receipt']['cycle_signatures']>0
 # a^n supplies distinct formal tapes for every n; finite alphabet repeats,
 # so this is never evidence for infinite nonrepeating coherent prose.

def test_actor_conflict_rejected():
 g=BilateralGrammar([Chunk('a','Eve, ','START','X','actor','fixture',(('actor','Eve'),)),Chunk('b','refer Eve.','X','END','actor','fixture',(('actor','Mom'),))]);s,_=g.extend(g.initial(),g.chunks[0],'left');n,error=g.extend(s,g.chunks[1],'right');assert n is None and error['reason']=='actor_conflict'

def test_paper_centerout_phrase_seam_uses_empty_debt_closure():
 from llm_palindrome.bilateral_seams import CenterOutGrammar
 cs=[Chunk('L1','Was it ','START','NP','question','control'),Chunk('L2','a ca','NP','CAR_SUFFIX','partial car','control'),Chunk('L3','r ','CAR_SUFFIX','OR_PREFIX','car completion','control'),Chunk('C','o','OR_PREFIX','OR_SUFFIX','partial or','connector lemma or'),Chunk('R3','r a ','OR_SUFFIX','CAT','or completion','control'),Chunk('R2','cat','CAT','REL','object noun','control'),Chunk('R1',' I saw?','REL','END','relative phrase','control',(('observer','I'),))]
 g=CenterOutGrammar(cs,['C']);s=g.initial()
 for pid,side in [('L3','left'),('R3','right'),('L2','left'),('R2','right'),('L1','left'),('R1','right')]:
  s,error=g.extend(s,next(c for c in cs if c.id==pid),side);assert error is None
 assert g.closed(s);assert not s['debt']['letters'];assert g.render(s)=='Was it a car or a cat I saw?';assert s['left_ids']==['L1','L2','L3'];assert s['right_ids']==['R3','R2','R1']
 assert len(g.search(max_steps=6)['outputs'])==1

def test_phrase_normalized_payoff_does_not_prefer_microscopic_chunks():
 from llm_palindrome.bilateral_seams import phrase_payoff_utility
 tiny=dict(paid=1,added=1);phrase=dict(paid=8,added=8)
 assert phrase_payoff_utility(phrase,'some men')>phrase_payoff_utility(tiny,'r')
 assert phrase_payoff_utility(dict(paid=0,added=8),'some men')==0

def test_empty_center_explicit_grammar_boundary():
 from llm_palindrome.bilateral_seams import CenterOutGrammar
 cs=[Chunk('l','no lemon','START','M','negative noun phrase','known control'),Chunk('r',', no melon.','M','END','negative noun phrase','known control')]
 r=CenterOutGrammar(cs,[],center_state='M').search(max_steps=2)
 assert r['outputs'][0]['text']=='no lemon, no melon.'
