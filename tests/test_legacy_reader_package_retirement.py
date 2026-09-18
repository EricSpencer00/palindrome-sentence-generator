from pathlib import Path

import pytest

from experiments.cross_boundary_material_probe import write_materials as write_cross_boundary_materials
from experiments.first_composition_rescue import write_materials as write_composition_materials


def test_legacy_reader_package_writers_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="retired"):
        write_composition_materials(tmp_path / "composition", [], 7, Path("data/v3_bank.json"))
    with pytest.raises(RuntimeError, match="retired"):
        write_cross_boundary_materials(tmp_path / "cross-boundary", {})


def test_existing_invalid_packages_are_withdrawn() -> None:
    root = Path("runs")
    withdrawn = (
        root / "first-composition-rescue-2026-09-12" / "rater-packet" / "WITHDRAWN-DO-NOT-DISTRIBUTE.md",
        root / "cross-boundary-material-probe-2026-09-12" / "blind-screen" / "WITHDRAWN-DO-NOT-DISTRIBUTE.md",
        root / "readability-length-study-2026-09-12" / "WITHDRAWN-DO-NOT-DISTRIBUTE.md",
        root / "revision-2026-09-07" / "WITHDRAWN-DO-NOT-DISTRIBUTE.md",
        root / "punct" / "WITHDRAWN-DO-NOT-DISTRIBUTE.md",
    )
    for marker in withdrawn:
        assert marker.is_file(), marker


def test_revision_participant_instructions_are_visibly_withdrawn() -> None:
    text = (Path("runs") / "revision-2026-09-07" / "HUMAN-INSTRUCTIONS.txt").read_text()
    assert text.startswith("WITHDRAWN")
