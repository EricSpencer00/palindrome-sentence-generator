from pathlib import Path

from tools.polaris.meaning_search_batch import REVISION, job_specs


def test_batch_has_bounded_replicated_symbolic_lanes_and_one_pinned_gpu_lane():
    specs = job_specs(Path("/tmp/meaning-results"), "/cluster/conda/python")
    assert len(specs) == 7
    assert len({row["id"] for row in specs}) == 7
    assert all(0 < row["hard_timeout_seconds"] <= 1200 for row in specs)
    assert sum(row["id"].startswith("connected-events") for row in specs) == 3
    assert sum(row["id"].startswith("typed-overhang") for row in specs) == 3
    model = specs[-1]
    assert model["command"][0] == "/cluster/conda/python"
    assert REVISION in model["command"]
    assert "--attempts" in model["command"]


def test_expanded_profile_raises_caps_and_reuses_only_selected_model_cache():
    specs = job_specs(Path("/tmp/meaning-results"), "/cluster/conda/python", "expanded", "/verified/task/cache")
    typed = next(s for s in specs if s["id"].startswith("typed-overhang"))
    event = next(s for s in specs if s["id"].startswith("connected-events"))
    assert ["--max-letters", "2500"] == typed["command"][typed["command"].index("--max-letters"):][:2]
    assert "--beam-width" in typed["command"]
    assert "--max-states" in event["command"]
    assert specs[-1]["command"][-2:] == ["--cache-dir", "/verified/task/cache"]
    assert len(specs) == 10
    assert sum(s["id"].startswith("free-typed") for s in specs) == 3
    assert max(s["hard_timeout_seconds"] for s in specs) < 600
