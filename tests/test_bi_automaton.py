from llm_palindrome.bi_automaton import Clause, ClauseAutomaton, intersect, tape

def test_independent_character_product_exact_svo():
    left=ClauseAutomaton([Clause("stressed deliver", "SVO")])
    right=ClauseAutomaton([Clause("reviled desserts", "SVO")], reverse=True)
    got=intersect(left,right)
    assert got and got[0]["exact"] and not got[0]["repair"]
    assert tape(got[0]["left"]+got[0]["right"]) == tape(got[0]["left"]+got[0]["right"])[::-1]

def test_no_aligned_boundary_shortcut():
    left=ClauseAutomaton([Clause("stressed deliver", "SVO")])
    right=ClauseAutomaton([Clause("reviled desserts", "SVO")], reverse=True)
    assert intersect(left,right)[0]["boundary_offsets"][0] != intersect(left,right)[0]["boundary_offsets"][1]
