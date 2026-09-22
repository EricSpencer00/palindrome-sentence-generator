import json

from experiments.build_seed_np_reader_package_20260922 import (
    RATER_COUNT, build, shuffled, tokens, write_package,
)
from experiments.seed_np_cross_role_intersection_20260922 import run


def _source(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps(run(max_rows=200)))
    return path


def test_shuffle_preserves_word_multiset_and_changes_order():
    text = "An aide rips nine memo-hero memos. Some more home men inspire Diana."
    changed = shuffled(text, seed=99)
    assert changed != text
    assert sorted(word.casefold() for word in tokens(changed)) == sorted(
        word.casefold() for word in tokens(text)
    )


def test_package_is_blinded_randomized_and_reproducible(tmp_path):
    source = _source(tmp_path)
    items, key = build(source)
    assert len(items) == len(key) == 5
    assert all(set(item) == {"opaque_id", "text"} for item in items)

    output = tmp_path / "study"
    result = write_package(output, source)
    assert result["raters"] == RATER_COUNT
    forms = sorted((output / "rater-package").glob("R*.json"))
    assert len(forms) == RATER_COUNT
    first = json.loads(forms[0].read_text())
    second = json.loads(forms[1].read_text())
    assert [row["opaque_id"] for row in first["items"]] != [
        row["opaque_id"] for row in second["items"]
    ]
    assert all("condition" not in row for row in first["items"])
    assert (output / "internal" / "analysis-plan.md").exists()
    assert (output / "MANIFEST-SHA256.json").exists()
