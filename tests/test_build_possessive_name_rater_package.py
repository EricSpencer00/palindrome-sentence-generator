import json

import pytest

from experiments.build_possessive_name_rater_package import build, shuffled
from experiments.possessive_name_relexicalizer import run


def source(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({
        "mechanically_admitted": [
            {"rendered": "Marge lets Hara see Sarah's telegram.", "checks": {"exact": True}},
            {"rendered": "Marge lets Aino see Sonia's telegram.", "checks": {"exact": True}},
        ]
    }))
    return path


def test_shuffle_preserves_words_but_changes_order():
    original = "Marge lets Hara see Sarah's telegram."
    changed = shuffled(original, seed=2026091203)
    assert changed != original
    assert sorted(changed.lower().replace(".", "").split()) == sorted(
        original.lower().replace(".", "").split()
    )


def test_retired_builder_refuses_any_source_not_only_the_known_ablation(tmp_path):
    with pytest.raises(RuntimeError, match="retired"):
        build(source(tmp_path))


def test_retired_builder_refuses_the_rejected_catalogue_family_run(tmp_path):
    path = tmp_path / "rejected.json"
    path.write_text(json.dumps(run()))
    with pytest.raises(RuntimeError, match="retired"):
        build(path)
