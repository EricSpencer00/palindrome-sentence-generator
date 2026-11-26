import ast
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from llm_palindrome.grammar_feedback import candidate, freeze_pairs, validate_feedback, append_feedback
from llm_palindrome.strict_selection import best_strict_search, strict_closure_gate
from llm_palindrome.search import WordTries, State, beam_search
from llm_palindrome.centerout import COState, centerout_search


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.items = [candidate('one', 'Some memos.', 'synthetic:test', 'l1'),
                      candidate('two', 'An aide rips nine memos; some men inspire Diana.', 'synthetic:test', 'l2')]
        self.packet = freeze_pairs(self.items, [('one', 'two')], 7)
        self.label = dict(packet_sha256=self.packet['packet_sha256'], pair_id='pair-001',
                          preference='neither', rater_kind='machine', rater_id='synthetic-test',
                          scores={c['id']: dict(grammar=0, meaning=0, rationale='Synthetic test label only.', flags=[])
                                  for c in self.items})

    def test_rejects_nonexact_unicode_tampered_or_heldout(self):
        for text in ['ordinary prose', 'Été', '']:
            with self.assertRaises(ValueError): candidate('bad', text, 'test', 'l1')
        bad = copy.deepcopy(self.packet); bad['candidates'][0]['text'] = 'Different text.'
        with self.assertRaises(ValueError): validate_feedback(bad, self.label)
        held = candidate('held', 'Some memos.', 'test', 'l3', split='held_out')
        with self.assertRaises(ValueError): freeze_pairs([held, self.items[1]], [('held','two')],7)

    def test_human_evidence_separation_and_duplicate_preservation(self):
        human = copy.deepcopy(self.label); human.update(rater_kind='human', rater_id='synthetic-human', raw_response='Neither.')
        with self.assertRaises(ValueError): validate_feedback(self.packet, human)
        validate_feedback(self.packet, human, human_response='Neither.')
        human.pop('scores')
        validate_feedback(self.packet, human, human_response='Neither.')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'labels.jsonl'
            append_feedback(path, self.packet, self.label)
            with self.assertRaises(ValueError): append_feedback(path, self.packet, self.label)
            self.assertEqual(len(path.read_text().splitlines()),1)

    def test_invalid_score_pair_and_deterministic_order(self):
        self.assertEqual(self.packet, freeze_pairs(self.items,[('one','two')],7))
        bad=copy.deepcopy(self.label); bad['scores']['one']['grammar']=True
        with self.assertRaises(ValueError): validate_feedback(self.packet,bad)
        bad=copy.deepcopy(self.label); bad['scores']['extra']=bad['scores']['one']
        with self.assertRaises(ValueError): validate_feedback(self.packet,bad)


class SelectionTests(unittest.TestCase):
    def test_higher_score_ineligible_closure_does_not_win(self):
        def fake_search(tries, scorer, *, allow_closed, **settings):
            for words in [('never','odd','or','even'), ('some','memos')]:
                if allow_closed(words,()): return list(words)
            return []
        self.assertEqual(best_strict_search(fake_search,None,None,min_letters=1,max_letters=100),['some','memos'])
        self.assertFalse(strict_closure_gate(1,100)(('ordinary','prose'),()))
        self.assertEqual(best_strict_search(fake_search,None,None,min_letters=1,max_letters=100,
                                          additional_gate=lambda l,r:False),[])

    def test_both_actual_search_apis_retain_eligible_synthetic_closure(self):
        class Zero:
            def word_delta(self,*args):return 0.0
        settings=dict(min_letters=1,max_letters=100,max_steps=1,diversity=0)
        out=best_strict_search(beam_search,WordTries([]),Zero(),
                              initial_state=State(0.0,('some',),('memos',),'m','L',0.0),**settings)
        self.assertEqual(out,['some','memos'])
        out=best_strict_search(centerout_search,WordTries([]),Zero(),
                              initial_state=COState(0.0,('some',),('memos',),'','L',0.0),**settings)
        self.assertEqual(out,['some','memos'])
        with self.assertRaises(ValueError): best_strict_search(centerout_search,None,None,center='a',**settings)

    def test_dependency_light_keys_match_frozen_source(self):
        from experiments.structural_checks import STRUCTURAL_CHECKS
        p=Path(__file__).resolve().parents[1]/'experiments/paired_directional_search_resource_sensitivity_20260925.py'
        tree=ast.parse(p.read_text())
        frozen=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='STRUCTURAL_CHECKS' for t in n.targets))
        self.assertEqual(STRUCTURAL_CHECKS,frozen)
        importlib.import_module('experiments.analyze_fragment_penalty_20261002')


if __name__=='__main__': unittest.main()
