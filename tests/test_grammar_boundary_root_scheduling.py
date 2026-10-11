import itertools
import unittest

from experiments.derived_paragraph_controls_20261009 import CANDIDATES, PATH, UNITS, tapes
from llm_palindrome.admission import normalize_letters
from llm_palindrome.block_search import BlockUnit, block_beam_search, compatible_actions
from llm_palindrome.block_seams import Seam
from llm_palindrome.grammar_boundaries import GrammarBoundaryIndex
from llm_palindrome.root_partition_search import root_partition_search
from llm_palindrome.typed_constituents import TypedGrammar, words
from experiments.block_seam_comparison_20261009 import render_paragraph


class BoundarySchedulingTests(unittest.TestCase):
    def test_character_prefixes_keep_partial_lexical_seams(self):
        grammar = TypedGrammar(set(words(CANDIDATES[1])))
        index = GrammarBoundaryIndex(grammar)
        for left, right in [('No ri', 'on'), ('Li', 'il'), ('No rider sees', 'sees red iron')]:
            self.assertTrue(index.allows_text(left, right))
        self.assertFalse(index.allows_text('the gardener', ''))
        self.assertFalse(index.allows_text('Nox', ''))
        self.assertTrue(index.allows_text('', ''))

    def test_both_supplied_cross_block_controls_keep_their_path(self):
        grammar = TypedGrammar(set(words(CANDIDATES[1])))
        index = GrammarBoundaryIndex(grammar)
        inventory = tuple(BlockUnit('reg-' + str(i), t) for i, t in enumerate(UNITS))
        def frontier(state):
            left, right = tapes(state)
            return grammar.paragraph_frontier(left, right, 4)
        state = Seam()
        states = []
        for side, text in PATH:
            menu = compatible_actions(state, inventory, grammar_accept=frontier,
                                      boundary_accept=index.allows_state)
            action = next(a for a in menu if a.side == side and a.unit.text == text)
            state = action.child
            states.append(state)
            self.assertTrue(index.allows_state(state))
        allowed = set(states)
        def close(state):
            left, right = tapes(state)
            parsed = grammar.paragraph(left + right, 4)
            return parsed is not None and 2 <= len(parsed) <= 4 and len({w for w,_ in parsed}) == len(parsed)
        result = block_beam_search(inventory, None,
            grammar_accept=lambda state: state in allowed and frontier(state),
            boundary_accept=index.allows_state,allow_closed=close,beam_width=64,
            max_steps=12,max_actions=10000,min_letters=30,max_letters=119,diversity=0)
        rendered = {render_paragraph(grammar.paragraph(sum(tapes(t['state']), ()), 4))
                    for t in result['terminals'] if t['eligible']}
        self.assertTrue(set(CANDIDATES) <= rendered)
        self.assertTrue(all(normalize_letters(text) == normalize_letters(text)[::-1]
                            for text in rendered))
        self.assertFalse(result['truncated'])

    def test_filtered_and_partitioned_small_language_match_exhaustive_terminals(self):
        class ToyGrammar:
            paths = [(((('a',), ('b',)), (('a',), ('b',)), (('a',), ('b',))), 'toy')]
        language = set(itertools.product(('a', 'b'), repeat=3))
        def frontier(state):
            left = tuple(p.text for p in state.left)
            right = tuple(p.text for p in state.right)
            return len(left)+len(right)<=3 and any(
                text[:len(left)] == left and (not right or text[-len(right):] == right)
                for text in language)
        def closed(state):
            return tuple(p.text for p in state.left+state.right) in language
        index = GrammarBoundaryIndex(ToyGrammar())
        settings=dict(grammar_accept=frontier,allow_closed=closed,beam_width=64,
                      max_steps=3,max_actions=10000,min_letters=3,max_letters=3,
                      max_words=3,diversity=0)
        reference=block_beam_search(['a','b'],None,**settings)
        filtered=block_beam_search(['a','b'],None,boundary_accept=index.allows_state,**settings)
        partitioned=root_partition_search(['a','b'],None,index,grammar_accept=frontier,
            allow_closed=closed,beam_width=64,max_steps=3,max_actions_per_root=10000,
            min_letters=3,max_letters=3,max_words=3,diversity=0,seconds_per_root=.5)
        expected={' '.join(t) for t in language if t==t[::-1]}
        for result in (reference,filtered):
            self.assertFalse(result['truncated'])
            self.assertEqual({t['state'].text() for t in result['terminals'] if t['eligible']},expected)
        terminals={t['state'].text() for lane in partitioned['lanes']
                   for t in lane['result']['terminals'] if t['eligible']}
        self.assertEqual(terminals,expected)
        self.assertEqual(partitioned['root_count'],4)
        self.assertEqual(partitioned['families_with_started_root'],partitioned['families_with_viable_root'])
        self.assertTrue(all(not lane['result']['truncated'] for lane in partitioned['lanes']))
        self.assertEqual([lane['seed'] for lane in partitioned['lanes']],[921,922,923,921])

    def test_boundary_rejections_have_their_own_reason(self):
        grammar=TypedGrammar(set(words(CANDIDATES[1])))
        index=GrammarBoundaryIndex(grammar)
        result=block_beam_search(['the gardener','no rider'],None,
            boundary_accept=index.allows_state,max_steps=1,min_letters=1,max_letters=119)
        rejected=[a for a in result['action_log'] if a['text']=='the gardener']
        self.assertEqual({a['reason'] for a in rejected},{'finite_grammar_boundary'})


if __name__ == '__main__':
    unittest.main()
