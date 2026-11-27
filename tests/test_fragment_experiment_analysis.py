import itertools
import json
import statistics
import unittest

from experiments.analyze_fragment_penalty_20261002 import analyze, validate_saved_row
from experiments.fragment_penalty_20261002 import OUT, tokens, trigrams


class FragmentAnalysisTests(unittest.TestCase):
    def test_saved_row_cross_checks_reject_tampered_outcomes(self):
        source = next(
            row for row in json.loads((OUT/'results.json').read_text())['results']
            if row['text']
        )
        manifest = json.loads((OUT/'manifest.json').read_text())
        job = next(job for job in manifest['jobs'] if job['id'] == source['id'])

        for field, value in (
            ('strict_admitted_selected_output', not source['strict_admitted_selected_output']),
            ('letters', source['letters'] + 1),
            ('word_count', source['word_count'] + 1),
        ):
            altered = json.loads(json.dumps(source))
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_saved_row(altered, job)

        altered = json.loads(json.dumps(source))
        altered['basic_structural_checks'].pop(next(iter(altered['basic_structural_checks'])))
        with self.assertRaises(AssertionError):
            validate_saved_row(altered, job)

        altered = json.loads(json.dumps(source))
        altered['basic_structural_checks']['exact_letter_palindrome'] = False
        with self.assertRaises(AssertionError):
            validate_saved_row(altered, job)

    def test_all_requested_jobs_and_independent_pairwise_recount(self):
        data = analyze(replicates=100)
        rows = json.loads((OUT/'results.json').read_text())['results']
        for penalty, expected_distinct, strict in [(0,69,59),(4,78,28),(16,78,28)]:
            arm = data['arms'][penalty]
            self.assertEqual((arm['requested'],arm['exact_outputs'],arm['distinct_texts'],arm['strict_outputs']), (120,90,expected_distinct,strict))
            sets = [set(trigrams(tokens(t))) for t in {r['text'] for r in rows if r['penalty']==penalty and r['text']}]
            independent = statistics.mean(len(a^b)/len(a|b) for a,b in itertools.combinations(sets,2))
            self.assertAlmostEqual(arm['mean_pairwise_trigram_jaccard_distance'], independent)
        self.assertEqual(data['paired_differences'][4]['exact_yield']['interval_95'],[0.0,0.0])
        self.assertEqual(data['arms'][4]['successful_records_with_targeted_fragments'],0)
        self.assertEqual(data['arms'][16]['successful_records_with_targeted_fragments'],0)


if __name__=='__main__':
    unittest.main()
