from experiments.controlled_pos_pruning import (
    _tasks,
    bootstrap_ratio_ci,
    results_markdown,
    summarize,
)


def row(seed, arm, accepted, generated, cpu):
    return {
        "seed": seed,
        "arm": arm,
        "vocabulary": 10,
        "stop_reason": "node_budget",
        "cpu_seconds": cpu,
        "wall_seconds": cpu,
        "peak_rss_mib": 100 + (arm == "incremental"),
        "peak_frontier": 10,
        "expansion_calls": 5,
        "candidate_expansions": generated + 2,
        "states_generated": generated,
        "states_pushed": generated,
        "states_popped": 100,
        "state_pruned": generated // 2 if arm == "incremental" else 0,
        "closed_states": accepted + 2,
        "eligible_closures": accepted + 1,
        "accepted": accepted,
    }


def test_summary_uses_paired_rate_ratios():
    rows = [
        row(0, "terminal", 2, 20, 4),
        row(0, "incremental", 6, 30, 3),
        row(1, "terminal", 4, 20, 4),
        row(1, "incremental", 12, 30, 3),
    ]
    result = summarize(rows, bootstrap_replicates=100)
    assert result["ratios"]["accepted_per_generated"]["incremental_over_terminal"] == 2
    assert result["ratios"]["accepted_per_popped"]["incremental_over_terminal"] == 3
    assert result["ratios"]["accepted_per_cpu_second"]["incremental_over_terminal"] == 4


def test_bootstrap_is_repeatable():
    rows = [
        row(0, "terminal", 2, 20, 4),
        row(0, "incremental", 5, 30, 3),
        row(1, "terminal", 4, 25, 5),
        row(1, "incremental", 9, 35, 4),
    ]
    assert bootstrap_ratio_ci(rows, "accepted", "states_generated", replicates=50) == \
        bootstrap_ratio_ci(rows, "accepted", "states_generated", replicates=50)


def test_summary_marks_zero_terminal_rate_as_not_estimable():
    rows = [
        row(0, "terminal", 0, 20, 4),
        row(0, "incremental", 1, 20, 4),
    ]
    result = summarize(rows, bootstrap_replicates=20)
    assert result["ratios"]["accepted_per_generated"]["incremental_over_terminal"] is None
    config = {
        "vocab": 10,
        "min_letters": 3,
        "max_letters": 8,
        "max_units": 6,
        "max_overhang": 4,
        "node_budget": 100,
    }
    assert "not estimable" in results_markdown(config, result)


def test_task_order_is_balanced_without_changing_pairs():
    tasks = _tasks([0, 1], {"node_budget": 10})
    assert [(seed, arm) for seed, arm, _, _ in tasks] == [
        (0, "terminal"), (0, "incremental"),
        (1, "incremental"), (1, "terminal"),
    ]


def test_task_order_pairs_each_opening():
    openings = [{"index": 4, "word": "a"}, {"index": 6, "word": "i"}]
    tasks = _tasks([0, 1], {"node_budget": 10}, openings)
    assert [opening["word"] for _, _, _, opening in tasks] == ["a", "a", "i", "i"]
