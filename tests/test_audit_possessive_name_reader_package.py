import json
from pathlib import Path

import pytest

from experiments.audit_possessive_name_reader_package import load_blinded_items, tokens, validate_controls
from experiments.build_possessive_name_rater_package import write_package


def source(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({
        "mechanically_admitted": [
            {"rendered": "Marge lets Hara see Sarah's telegram.", "checks": {"exact": True}},
            {"rendered": "Marge lets Aino see Sonia's telegram.", "checks": {"exact": True}},
        ]
    }))
    return path


def test_retired_builder_cannot_create_new_auditable_package(tmp_path):
    with pytest.raises(RuntimeError, match="retired"):
        write_package(tmp_path / "package", source(tmp_path))
    assert tokens("Sarah's telegram.") == ["sarahs", "telegram"]


def test_audit_rejects_a_shuffle_that_keeps_the_candidate_order(tmp_path):
    package = tmp_path / "package"
    (package / "internal").mkdir(parents=True)
    (package / "rater-package").mkdir()
    (package / "internal" / "condition-key.json").write_text(json.dumps([
        {"opaque_id": "A", "block": "B", "condition": "candidate"},
        {"opaque_id": "B", "block": "B", "condition": "intact_prose"},
        {"opaque_id": "C", "block": "B", "condition": "word_shuffle"},
    ]))
    (package / "rater-package" / "R001.json").write_text(json.dumps({"items": [
        {"opaque_id": "A", "text": "Marge lets Hara see Sarah's telegram."},
        {"opaque_id": "B", "text": "Marge lets Hara read Sarah's telegram."},
        {"opaque_id": "C", "text": "Marge lets Hara see Sarah's telegram."},
    ]}))
    key, blinded = load_blinded_items(package)
    try:
        validate_controls(key, blinded)
    except ValueError as exc:
        assert "shuffle" in str(exc)
    else:
        raise AssertionError("broken control must be rejected")


def test_every_historical_distribution_folder_is_marked_withdrawn():
    root = Path(__file__).resolve().parents[1]
    for package in (
        root / "runs/possessive-name-reader-package-2026-09-12",
        root / "runs/possessive-name-reader-package-2026-09-12-repro",
    ):
        for folder in (package, package / "rater-package", package / "internal"):
            marker = folder / "WITHDRAWN-DO-NOT-DISTRIBUTE.md"
            assert marker.is_file(), marker
            assert "Do not" in marker.read_text() or "do not" in marker.read_text()
