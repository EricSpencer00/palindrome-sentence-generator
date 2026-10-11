import itertools
from llm_palindrome.typed_constituents import TypedGrammar
from llm_palindrome.block_search import block_beam_search
from experiments.derived_paragraph_controls_20261009 import tapes

def toy():
 g=TypedGrammar([]);g.paths=[(((('a',),('b',)),)*3,'toy')];g.path_metadata=[{'tense':'present'}];return g

def test_positions_keys_reverse_boundary_and_mutation():
 g=toy();slots=g.paths[0][0];g._cached_positions.cache_clear()
 for n in range(7):
  for t in itertools.product(('a','b'),repeat=n):
   for reverse in (True,False):assert g._positions(slots,t,reverse)==g._positions_uncached(slots,t,reverse)
 first=g._positions(slots,('a',));first.clear();assert g._positions(slots,('a',))
 assert g._positions(slots,('ab',))==[]
 different=(((('ab',),),),)
 # Exact slot contents, lexical seams and reverse flag are independent keys.
 for i in range(8200):g._positions(slots,(str(i),))
 assert g.positions_cache_info().currsize<=8192

def test_complete_toy_terminal_set_cached_and_uncached(monkeypatch):
 expected={' '.join(t) for n in (3,6) for t in itertools.product(('a','b'),repeat=n) if t==t[::-1]}
 def search(g):
  return block_beam_search(['a','b'],None,grammar_accept=lambda s:g.paragraph_frontier(*tapes(s),2),allow_closed=lambda s:g.paragraph(sum(tapes(s),()),2) is not None,beam_width=512,max_steps=6,max_actions=100000,min_letters=3,max_letters=6,max_words=6,diversity=0,seed=921)
 cached=search(toy())
 monkeypatch.setattr(TypedGrammar,'_positions',staticmethod(TypedGrammar._positions_uncached))
 uncached=search(toy())
 for result in (cached,uncached):
  assert not result['truncated'];assert {t['state'].text() for t in result['terminals'] if t['eligible']}==expected
 assert cached['attempted_actions']==uncached['attempted_actions']
 assert cached['visited_states']==uncached['visited_states']
