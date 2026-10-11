import itertools
import pytest
from llm_palindrome.paragraph_residual_join import exact_residual_pairs,JoinBudgetExceeded

def test_unequal_halves_and_occurrence_duplicates_match_exhaustive():
 strings=['']+[''.join(t) for n in range(1,5) for t in itertools.product('ab',repeat=n)]
 left=strings+['ab'];right=list(reversed(strings))+['ba']
 expected={(i,j) for i,l in enumerate(left) for j,r in enumerate(right) if l+r==(l+r)[::-1]}
 actual,receipt=exact_residual_pairs(left,right)
 assert set(actual)==expected and len(actual)==len(expected)
 assert receipt['complete']
 for l,r in [('ababa','ba'),('ab','ababa'),('noriderseesmail','liamseesrediron')]:
  pairs,_=exact_residual_pairs([l],[r]);assert bool(pairs)==(l+r==(l+r)[::-1])

def test_budget_cannot_be_reported_as_complete():
 with pytest.raises(JoinBudgetExceeded):exact_residual_pairs(['aaa'],['aaa'],max_work=1)
