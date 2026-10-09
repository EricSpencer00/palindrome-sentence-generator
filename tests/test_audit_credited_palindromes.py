import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.audit_credited_palindromes import audit, credited_markdown, mirrored_word_spans


def post(text, author="u/alice"):
    return {"text": text, "author": author,
            "source_url": f"https://www.reddit.com/r/palindromes/comments/{author[2:]}/example/",
            "author_url": f"https://www.reddit.com/user/{author[2:]}/"}


def test_exact_tape_dedup_keeps_each_original_credit():
    rows = [post("A man, a plan, a canal: Panama!"),
            post("a man a plan a canal panama", "u/bob")]
    result = audit({"collected_on": "2026-10-09", "rows": rows})
    assert result["summary"]["exact_posts"] == 2
    assert result["summary"]["unique_exact"] == 1
    assert result["summary"]["duplicate_exact_posts"] == 1
    assert result["exact"][0]["sources"] == rows
    assert result["metadata"] == {"collected_on": "2026-10-09"}
    markdown = credited_markdown(result)
    assert "u/alice" in markdown and "u/bob" in markdown
    assert all(row["source_url"] in markdown for row in rows)


def test_non_palindromes_empty_tapes_and_non_ascii_are_separate():
    result = audit([post("Madam? I saw an Adam."), post("123?!"),
                    post("éve"), post("Kayak"), post("Madam?")])
    assert result["summary"]["exact_posts"] == 1
    assert result["summary"]["non_palindromes"] == 2
    assert result["summary"]["unsupported"] == 2
    assert result["non_palindromes"][0]["first_mismatch"]["left"] != result["non_palindromes"][0]["first_mismatch"]["right"]
    assert result["non_palindromes"][1]["reason"] == "empty_letter_tape"
    assert result["unsupported"][0]["unsupported_letters"] == ["é"]


def test_mirrored_spans_cover_shifted_word_boundaries_exactly():
    # Panama's six letters cover four opposite words: a/man/a/p (from plan).
    spans = mirrored_word_spans("A man, a plan, a canal: Panama!")
    panama = spans[-1]
    assert panama["span"] == [15, 21]
    assert panama["mirrored_span"] == [0, 6]
    assert panama["fanout"] == 4
    assert [x["word"] for x in panama["overlaps"]] == ["A", "man", "a", "plan"]
    assert [x["letters"] for x in panama["overlaps"]] == [1, 3, 1, 1]
    assert all(sum(x["letters"] for x in word["overlaps"]) == word["letters"] for word in spans)


def test_smart_apostrophe_stays_inside_word_and_center_can_overlap_itself():
    result = audit([post("Madam, I’m Adam.")])
    assert result["summary"]["unique_exact"] == 1
    spans = result["exact"][0]["word_spans"]
    assert [x["word"] for x in spans] == ["Madam", "I’m", "Adam"]
    assert spans[1]["span"] == [5, 7]
    assert any(x["word_index"] == 1 for x in spans[1]["overlaps"])
    assert mirrored_word_spans("level")[0]["fanout"] == 1


def test_touching_boundaries_do_not_count_as_overlap():
    spans = mirrored_word_spans("dog god")
    assert spans[0]["fanout"] == 1
    assert spans[0]["overlaps"][0]["word"] == "god"
    assert spans[1]["overlaps"][0]["word"] == "dog"


def test_invalid_schema_fails_and_empty_corpus_has_no_top_example():
    with pytest.raises(ValueError, match="requires string fields"):
        audit([{"text": "eve"}])
    with pytest.raises(ValueError, match="rows list"):
        audit({"rows": "wrong"})
    assert audit([])["summary"]["top_fanout_example"] is None


@pytest.mark.parametrize("field,value", [
    ("author", ""), ("author", " \t"), ("source_url", ""),
    ("source_url", "javascript:alert(1)"), ("source_url", "https:///missing-host"),
    ("author_url", " "), ("author_url", "https://www.reddit.com/user/a b"),
    ("author_url", "https://[broken"), ("author_url", "https://example.com:invalid/"),
])
def test_missing_or_invalid_credit_cannot_be_admitted(field, value):
    row = post("eve")
    row[field] = value
    with pytest.raises(ValueError, match="requires"):
        audit([row])


def test_markdown_keeps_original_titles_and_credits_for_every_bucket():
    rows = [post("Never odd or even.", "u/exact"),
            post("Madam? I saw an Adam.", "u/nonpalindrome"),
            post("éve", "u/unsupported")]
    markdown = credited_markdown(audit(rows))
    assert "## Verified exact palindromes" in markdown
    assert "## Excluded: not exact" in markdown
    assert "## Unsupported alphabetic characters" in markdown
    for row in rows:
        assert row["text"] in markdown
        assert row["author"] in markdown
        assert row["source_url"] in markdown
        assert row["author_url"] in markdown
    assert markdown.index(rows[1]["text"]) > markdown.index("## Excluded: not exact")
    assert markdown.index(rows[2]["text"]) > markdown.index("## Unsupported alphabetic characters")


def test_offline_cli_writes_auditable_files(tmp_path):
    source = tmp_path / "posts.json"
    source.write_text(json.dumps({"rows": [post("Never odd or even.")]}))
    output = tmp_path / "audit"
    script = Path(__file__).resolve().parents[1] / "tools" / "audit_credited_palindromes.py"
    subprocess.run([sys.executable, str(script), str(source), "--out", str(output)], check=True, capture_output=True, text=True)
    assert json.loads((output / "audit.json").read_text())["summary"]["unique_exact"] == 1
    assert "Reddit post" in (output / "CREDITS.md").read_text()
