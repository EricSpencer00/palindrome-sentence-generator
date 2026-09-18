import pytest

from experiments.analyze_readability_length_study import analyze
from experiments.freeze_readability_length_study import study_rows, system_item


def test_retired_length_study_cannot_build_or_analyze_legacy_material(tmp_path):
    for call in (
        lambda: system_item("short", 80, 0),
        lambda: study_rows(1, 0),
        lambda: analyze(tmp_path),
    ):
        with pytest.raises(RuntimeError, match="retired"):
            call()
