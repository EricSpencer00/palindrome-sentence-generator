from llm_palindrome.fragment_frontier import FragmentFrontier

def test_partial_words_preserve_their_role_and_do_not_admit_invalid_completions():
 f=FragmentFrontier([{'text':'I open mail.','tokens':['i','open','mail'],'roles':['agent','verb','theme'],'frame':'open'}])
 p=f.positions('I op')[0]
 assert p.expected_word=='open' and p.consumed_in_word==2 and p.remaining_in_word=='en' and p.argument_role=='verb'
 assert f.positions('I open m')[0].argument_role=='theme'
 assert f.complete('I open mail.') and not f.complete('I op')
 assert not f.positions('I option') and not f.positions('I open association')
 assert f.positions('I op')==f.positions('I op')
