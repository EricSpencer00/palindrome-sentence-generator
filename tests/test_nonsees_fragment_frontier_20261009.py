import itertools
import pytest
from experiments.nonsees_fragment_frontier_20261009 import bank,cuts
from llm_palindrome.fragment_frontier import FragmentFrontier
from llm_palindrome.admission import normalize_letters as norm
from llm_palindrome.paragraph_residual_join import exact_residual_pairs

def test_malformed_licensed_paths_fail_closed():
    for entry in [
        dict(text='I open mail.',tokens=['I','open','mail'],roles=['agent','verb'],frame='open'),
        dict(text='I open mail.',tokens=['I','open','gate'],roles=['agent','verb','theme'],frame='open'),
        dict(text='I open mail.',tokens=['I','','open','mail'],roles=['agent','bad','verb','theme'],frame='open'),
    ]:
        with pytest.raises(ValueError):FragmentFrontier([entry])

def test_licensed_roles_agreement_and_partial_offsets():
    entries=bank();f=FragmentFrontier(entries)
    assert not f.complete('Anna open mail.')
    assert not f.complete('Otto pets mail.')
    assert not f.complete('Let mail in.')
    assert not f.complete('Ma is association as.')
    assert not f.complete('I open association as.')
    for e in entries:
        for at,role in cuts(e,'partial_word'):
            assert any(p.argument_role==role and not p.complete for p in f.positions(norm(e['text'])[:at]))

def test_nonsees_fixture_residual_outputs_equal_exhaustive_and_keep_duplicates():
    entries=[e for e in bank() if e['frame']=='step' or (e['frame']=='comparative' and e['tokens'][0]=='Ma')]
    tapes=[norm(e['text']) for e in entries]
    for n in (2,3):
        l=list(itertools.product(range(len(entries)),repeat=n//2));r=list(itertools.product(range(len(entries)),repeat=n-n//2))
        lt=[''.join(tapes[i] for i in s) for s in l];rt=[''.join(tapes[i] for i in s) for s in r]
        got,_=exact_residual_pairs(lt,rt)
        assert set(got)=={(i,j) for i,a in enumerate(lt) for j,b in enumerate(rt) if a+b==(a+b)[::-1]}
        assert len(got)==len(set(got))
