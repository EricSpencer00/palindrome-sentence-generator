"""Offline verification of the completed low-penalty development probe."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks


def main():
    out = ROOT / "runs/fragment-penalty-tuning-20261008"
    manifest_bytes = (out / "manifest-v4.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    result = json.loads((out / "results-v3.json").read_text())
    assert manifest_bytes == (out / "manifest-v4-executed.json").read_bytes()
    assert result["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert result["status"] == "complete"
    jobs = {job["id"]: job for job in manifest["jobs"]}
    rows = result["results"]
    assert len(rows) == len(jobs) == 32
    assert {row["id"] for row in rows} == set(jobs)
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    selected = []
    for row in rows:
        assert all(row[key] == value for key, value in jobs[row["id"]].items())
        assert row["status"] in {"deadline_best_so_far_output", "deadline_no_valid_output",
                                  "selected_basic_valid_output", "no_basic_valid_output"}
        if row["text"]:
            tape = "".join(c.lower() for c in row["text"] if c.isascii() and c.isalpha())
            assert tape and tape == tape[::-1]
            assert len(tape) == row["letters"]
            assert row["min_letters"] <= len(tape) <= row["max_letters"]
            assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"]
            checks = mechanical_admission_checks(row["text"], min_letters=row["min_letters"],
                                                  max_letters=row["max_letters"])
            assert checks == row["checks"]
            assert row["strict_admitted_selected_output"] == all(checks.values())
            assert len(row["text"].rstrip(".").split()) == row["word_count"] <= 256
            assert row["independently_exact"] is True
            selected.append(row)
        else:
            assert row["letters"] == row["word_count"] == 0
            assert not row["checks"] and row["strict_admitted_selected_output"] is None
    assert len(selected) == 16
    assert max(row["letters"] for row in selected) == 824
    assert sum(row["strict_admitted_selected_output"] for row in selected) == 0
    for penalty in (0, 1, 2, 4):
        assert sum(row["penalty"] == penalty for row in rows) == 8
        assert sum(row["penalty"] == penalty for row in selected) == 4
    print("Verified 32 jobs, 16 exact outputs, complete structural maps, and 824-letter maximum observed")


if __name__ == "__main__":
    main()
