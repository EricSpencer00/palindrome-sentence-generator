import itertools
from llm_palindrome.typed_constituents import TypedGrammar,words
from experiments.derived_paragraph_controls_20261009 import CANDIDATES

def test_exact_keys_and_uncached_equivalence():
 g=TypedGrammar(set(words(CANDIDATES[1])))
 g._cached_paragraph_frontier.cache_clear()
 tapes=[(),('no',),('no','rider'),('norider',),('sees',),('liam',),('mail',),('red','iron'),('no','rider','sees','mail')]
 for l,r,b in itertools.product(tapes,tapes,range(1,5)):
  assert g.paragraph_frontier(l,r,b)==g._paragraph_frontier_uncached(l,r,b)
 before=g.frontier_cache_info()
 assert g.paragraph_frontier(('no','rider'),(),max_sentences=4)
 assert not g.paragraph_frontier(('norider',),(),4)
 assert g.frontier_cache_info().hits==before.hits+2

def test_grammar_identity_budget_and_bound():
 g=TypedGrammar(set(words(CANDIDATES[1])));empty=TypedGrammar([])
 assert g.paragraph_frontier(('liam',),(),1)
 assert not empty.paragraph_frontier(('liam',),(),1)
 l=words('No rider sees mail Leon sees red iron');r=words('No rider sees Noel Liam sees red iron')
 assert not g.paragraph_frontier(l,r,3)
 assert g.paragraph_frontier(l,r,4)
 for i in range(4100):assert not empty.paragraph_frontier((str(i),),(),1)
 assert empty.frontier_cache_info().currsize<=4096

def test_exceptions_are_not_cached(monkeypatch):
 g=TypedGrammar([]);g._cached_paragraph_frontier.cache_clear();calls=[]
 def flaky(l,r,b):
  calls.append(1)
  if len(calls)==1:raise RuntimeError('interrupted')
  return False
 monkeypatch.setattr(g,'_paragraph_frontier_uncached',flaky)
 import pytest
 with pytest.raises(RuntimeError):g.paragraph_frontier(('x',),(),2)
 assert not g.paragraph_frontier(('x',),(),2)
 assert not g.paragraph_frontier(('x',),(),2)
 assert len(calls)==2
