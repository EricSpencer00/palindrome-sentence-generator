import itertools
import time
import unittest
from llm_palindrome.block_seams import Piece,Seam
from llm_palindrome.typed_constituents import TypedGrammar,words


class BidirectionalBlockSearchTests(unittest.TestCase):
    def test_initial_state_must_fit_resource_bounds(self):
        from llm_palindrome.block_search import block_beam_search
        state=Seam((Piece('seed',0,'a a'),),())
        for bounds in ({'max_words':1},{'max_letters':1}):
            with self.assertRaisesRegex(ValueError,'initial state exceeds'):
                block_beam_search(['a'],None,initial_state=state,max_steps=0,**bounds)

    def test_same_side_grammar_actions_reach_actual_search_menu(self):
        from llm_palindrome.block_search import compatible_actions,block_beam_search
        inventory=['a','a pot','a tray','no','the','the map']
        vocabulary=set(words('Eva carries can see a cave a pot a tray no the map'))
        g=TypedGrammar(vocabulary)
        s=Seam((Piece('subject',0,'Eva carries'),),(Piece('object',0,'a cave'),))
        self.assertEqual(s.debt()['residual'],'rries')
        actions=compatible_actions(s,inventory,grammar_accept=lambda c:g.paragraph_frontier(words(' '.join(p.text for p in c.left)),words(' '.join(p.text for p in c.right)),4))
        self.assertEqual({a.unit.text for a in actions if a.side=='left'},set(inventory))
        self.assertTrue(all(a.child.debt()['viable'] for a in actions))
        result=block_beam_search(inventory,None,initial_state=s,max_steps=1,max_actions=100,
            grammar_accept=lambda c:g.paragraph_frontier(words(' '.join(p.text for p in c.left)),words(' '.join(p.text for p in c.right)),4))
        self.assertEqual({x['text'] for x in result['action_log'] if x['side']=='left' and x['status']=='retained_proposal'},set(inventory))

    def test_exhaustive_terminal_sets_match_reference(self):
        from llm_palindrome.block_search import block_beam_search
        inventory=['a','b','ab','ba'];steps=3
        for l,r in itertools.product(['','a','b','ab'],repeat=2):
            s=Seam((Piece('seedL',0,l),) if l else (), (Piece('seedR',0,r),) if r else ())
            if not s.debt()['viable']:continue
            frontier={s};expected=set()
            for depth in range(steps+1):
                for p in frontier:
                    if p.exact():expected.add(p.text())
                if depth==steps:break
                nxt=set()
                for p in frontier:
                    for side,w in itertools.product(('left','right'),inventory):
                        # Reference tests viability using direct full tape prefixes.
                        left=' '.join(x.text for x in p.left)
                        right=' '.join(x.text for x in p.right)
                        a=(left+' '+w if side=='left' else left).replace(' ','')
                        b=(w+' '+right if side=='right' else right).replace(' ','')[::-1]
                        if not (a.startswith(b) or b.startswith(a)):continue
                        child=Seam(p.left+(Piece(w,0,w),),p.right) if side=='left' else Seam(p.left,(Piece(w,0,w),)+p.right)
                        nxt.add(child)
                frontier=nxt
            result=block_beam_search(inventory,None,initial_state=s,max_steps=steps,
                beam_width=10000,max_actions=100000,max_letters=30,max_words=20,diversity=0)
            self.assertFalse(result['truncated'])
            self.assertEqual({x['state'].text() for x in result['terminals']},expected,(l,r))

    def test_limits_and_deadline_are_explicit(self):
        from llm_palindrome.block_search import block_beam_search
        r=block_beam_search(['a','b'],None,max_steps=5,max_actions=1,beam_width=8)
        self.assertEqual(r['status'],'action_cap')
        self.assertLessEqual(r['attempted_actions'],1)
        r=block_beam_search(['a'],None,max_steps=5,deadline=time.monotonic()-1)
        self.assertIn(r['status'],('deadline','hard_deadline'))
        self.assertEqual(r['attempted_actions'],0)

    def test_callback_cannot_overrun_deadline_unboundedly(self):
        from llm_palindrome.block_search import block_beam_search
        def stalled(state):
            while True:pass
        start=time.monotonic()
        r=block_beam_search(['a'],None,grammar_accept=stalled,deadline=start+.02)
        self.assertEqual(r['status'],'hard_deadline')
        self.assertLess(time.monotonic()-start,.5)
        self.assertEqual(r['action_log'][0]['status'],'interrupted')

    def test_beam_pruning_is_reported(self):
        from llm_palindrome.block_search import block_beam_search
        r=block_beam_search(['a','b'],None,beam_width=1,max_steps=2,diversity=.4)
        self.assertTrue(r['truncated'])
        self.assertTrue(any(x['status']=='beam_pruned' for x in r['action_log']))

    def test_word_and_letter_bounds_precede_grammar(self):
        from llm_palindrome.block_search import block_beam_search
        calls=[]
        r=block_beam_search(['a a','abab'],None,max_steps=1,max_letters=3,max_words=1,
            grammar_accept=lambda s:calls.append(s) or True)
        self.assertFalse(calls)
        self.assertEqual({x['reason'] for x in r['action_log']},{'word_cap','letter_cap'})

    def test_frontier_bucket_supports_structured_grammar_metadata(self):
        from llm_palindrome.block_search import block_beam_search,compatible_actions
        r=block_beam_search(['a','b'],None,max_steps=1,frontier_key=lambda s:{'role':'NP'})
        self.assertTrue(r['action_log'])
        self.assertEqual(compatible_actions(Seam(),['a a','abab'],max_letters=3,max_words=1),[])


if __name__=='__main__':unittest.main()
