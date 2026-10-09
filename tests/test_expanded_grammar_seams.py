import unittest
from experiments.expanded_grammar_seams_20261009 import grammar,Stream,inventory,terminal,run
from llm_palindrome.admission import normalize_letters


class ExpandedTests(unittest.TestCase):
    def test_agreement_valency_and_question_forms(self):
        paths=grammar()
        self.assertTrue(any(p['family']=='intransitive' for p in paths))
        self.assertTrue(any(p['family']=='aux-question' and p['question'] for p in paths))
        for p in paths:
            self.assertEqual(len(p['slots']),len(p['roles']))
            if p['family'].startswith('SVO-present-sg'):
                self.assertNotIn('I',p['slots'][0]);self.assertNotIn('We',p['slots'][0])
            if p['family']=='copular-adjective' and p['slots'][1]==('am',):
                self.assertEqual(p['slots'][0],('I',))

    def test_reverse_grammar_completion_cannot_repair_a_word(self):
        paths=grammar();tid=next(i for i,p in enumerate(paths) if p['family']=='vocative-question')
        stream=Stream(((tid,0),))
        for word in ('cave','a','in','bees','see','I','can','Eva,'):
            options=inventory(paths,stream,'right');self.assertIn(word,options)
            stream=Stream(tuple(options[word]),stream.active+(word,))
        self.assertEqual(terminal(paths,stream),(tid,))
        self.assertEqual(normalize_letters(' '.join(stream.active[::-1])),'evacaniseebeesinacave')

    def test_known_control_and_bounded_status(self):
        r=run(seconds=.02,max_states=20)
        self.assertLessEqual(r['states_visited'],20)
        for c in r['known_controls']:
            self.assertTrue(c['not_novel']);t=normalize_letters(c['text']);self.assertEqual(t,t[::-1])
        self.assertIsNone(r['human_ratings'])


if __name__=='__main__':unittest.main()
