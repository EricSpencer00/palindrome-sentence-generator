"""Verify the figure's new summaries and its treatment of duplicate records."""
from collections import defaultdict
import json
from statistics import mean
import unittest

from paper.build_diagnostic_figures import ROOT, collect


class DiagnosticFigureTests(unittest.TestCase):
    def test_saved_plot_data_reproduces_with_all_controls(self):
        data = collect()
        self.assertEqual(data, json.loads((ROOT / "paper/fig/diagnostic-data.json").read_bytes()))
        stats = data["statistics"]
        self.assertEqual(stats["opposite_sign_distinct_texts"], 28)
        self.assertEqual(stats["shuffle_ranges_crossing_zero"], 63)
        self.assertEqual(stats["candidate_above_all_32_shuffles"], 2)
        self.assertEqual(stats["candidate_below_all_32_shuffles"], 1)
        self.assertEqual(stats["frequent_trigrams"][0], ["no it a", 46])
        self.assertTrue(all(len(r["gpt2_candidate_minus_each_shuffle"]) == 32 for r in data["records"]))

    def test_groups_give_equal_weight_to_distinct_texts(self):
        data = collect()
        by_text = defaultdict(list)
        for row in data["records"]:
            by_text[row["text_sha256"]].append(row)
        for row in data["distinct_texts"]:
            copies = by_text[row["text_sha256"]]
            self.assertEqual(row["record_count"], len(copies))
            self.assertEqual(row["brown_gain"], mean(x["brown_gain"] for x in copies))
        groups = data["statistics"]["phrase_groups"]
        self.assertEqual(groups["phrase_present"]["distinct_texts"], 24)
        self.assertEqual(groups["phrase_absent"]["distinct_texts"], 28)
        self.assertAlmostEqual(groups["phrase_present"]["mean_brown_gain"], .20061072527842325)
        self.assertAlmostEqual(groups["phrase_present"]["mean_gpt2_gain"], -.012503217617263432)


if __name__ == "__main__":
    unittest.main()
