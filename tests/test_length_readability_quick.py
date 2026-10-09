import json
import time

import pytest

from tools.polaris.length_readability_quick import (
    BANDS, CONFIG, CountedRankScorer, exact_record, failure_row, make_jobs,
    run_job, run_jobs, unit_counts, within_caps,
)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.search import WordTries


def test_grid_is_fully_paired_and_reproducible():
    jobs = make_jobs()
    assert jobs == make_jobs()
    assert len(jobs) == len({row["id"] for row in jobs}) == 72
    for low, high in BANDS:
        band = [row for row in jobs if row["min_letters"] == low]
        assert len(band) == 12
        assert {row["max_letters"] for row in band} == {high}
        for seed in {row["seed"] for row in band}:
            assert len([row for row in band if row["seed"] == seed]) == 4


def test_phrase_caps_count_letters_without_spaces_and_all_words():
    units = ("put rats", "at", "a", "startup")
    assert unit_counts(units) == {"letters": 17, "words": 5, "units": 4, "phrase_units": 1}
    assert within_caps(units, 17, 5)
    assert not within_caps(units, 16, 5)
    assert not within_caps(units, 17, 4)
    result = exact_record(units)
    assert result["independent_exact"]
    assert result["text"] == "Put rats at a startup."


def test_phrase_reuse_cannot_hide_constituent_repetition():
    words = ["a", "cat"]
    zero = CountedRankScorer(words, reuse_weight=0, order_weight=0, length_weight=0)
    penalized = CountedRankScorer(words, reuse_weight=-2, order_weight=0, length_weight=0)
    args = (("a cat", "a cat"), (), "L", "a cat", "append")
    assert penalized.word_delta(*args) - zero.word_delta(*args) == pytest.approx(-4)


@pytest.mark.parametrize("direction", ["outside_in", "center_out"])
def test_tiny_search_respects_exactness_and_length_band(direction):
    words = ["a", "cat", "tac"]
    resources = {"words": words, "bigrams": BigramModel({}, {}),
                 "tries": {"words": WordTries(words)}}
    config = {**CONFIG, "max_steps": 8, "beam_width": 4, "candidate_limit": 8,
              "per_parent": 2, "search_seconds": 0.1, "strict_max_words": 0}
    job = {"id": "tiny", "direction": direction, "inventory": "words", "seed": 1,
           "min_letters": 2, "max_letters": 9}
    result = run_job(job, resources, config)
    assert result["status"] in {"completed_with_output", "deadline_with_output"}
    assert result["selected"]["independent_exact"]
    assert 2 <= result["selected"]["letters"] <= 9
    assert result["strict_status"] == "not_run_word_limit"


def _slow_fixture(connection, job, resources, config):
    connection.send(("progress", {"longest_exact": {"text": "Eve.", "letters": 3, "independent_exact": True}}))
    time.sleep(2)


def _error_fixture(connection, job, resources, config):
    connection.send(("result", failure_row(job, "execution_error", 0.01, "fixture failure")))
    connection.close()


def test_supervisor_stops_hung_job_and_preserves_partial_output(tmp_path):
    job = {"id": "timeout-fixture"}
    rows = run_jobs([job], {}, {"hard_seconds": 0.15}, 1, tmp_path, child_target=_slow_fixture)
    assert rows[0]["status"] == "hard_deadline"
    assert rows[0]["longest_exact"]["text"] == "Eve."
    assert rows[0]["strict_admitted"] is None
    assert json.loads((tmp_path / "progress.json").read_text())["completed"] == 1


def test_supervisor_retains_failed_job_without_fabricating_scores(tmp_path):
    rows = run_jobs([{"id": "error-fixture"}], {}, {"hard_seconds": 1}, 1, tmp_path, child_target=_error_fixture)
    assert rows[0]["status"] == "execution_error"
    assert rows[0]["selected"] is None
    assert rows[0]["error"] == "fixture failure"
    assert "readability_score" not in rows[0]
