"""Independently verify the saved exploratory length probe without inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/length-cap-probe-20261008"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "results.json").read_text())
    assert result["manifest_sha256"] == sha((OUT / "manifest.json").read_bytes())
    assert result["status"] == "complete"
    jobs = {job["id"]: job for job in manifest["jobs"]}
    rows = result["results"]
    assert len(rows) == len(jobs) == 16
    assert {row["id"] for row in rows} == set(jobs)
    vocabulary = manifest["vocabulary"]["words"]
    assert sha(("\n".join(vocabulary) + "\n").encode()) == manifest["vocabulary"]["filtered_word_sha256"]
    maximum_word = max(map(len, vocabulary))
    groups = {}
    for row in rows:
        assert all(row[key] == value for key, value in jobs[row["id"]].items())
        assert row["status"] in {"output", "no_output"}
        counts = groups.setdefault(f"{row['minimum']}-{row['maximum']}/cap{row['max_words']}", Counter())
        counts["jobs"] += 1
        counts["outputs"] += row["status"] == "output"
        counts["strict_passes"] += row["strict_admitted"]
        counts["deadlines"] += row["deadline_reached"]
        if row["text"]:
            tape = "".join(c.lower() for c in row["text"] if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            assert tape and tape == tape[::-1]
            assert len(tape) == row["letters"]
            assert row["minimum"] <= len(tape) <= row["maximum"]
            words = row["text"].rstrip(".").lower().split()
            assert len(words) == row["words"] <= row["max_words"]
            assert all(word in vocabulary for word in words)
            assert sha(tape.encode()) == row["normalized_sha256"]
            assert sha(row["text"].encode()) == row["text_sha256"]
            assert row["strict_admitted"] == all(row["checks"].values())
            assert row["independently_exact"]
        else:
            assert row["letters"] == row["words"] == 0
            assert not row["checks"] and not row["strict_admitted"]
    continuation = result["continuation"]
    assert continuation["initial_results_sha256"] == sha((OUT / "results-interrupted-1.json").read_bytes())
    assert set(continuation["restarted_jobs"]).isdisjoint(continuation["completed_before_interruption"])
    assert set(continuation["restarted_jobs"]) | set(continuation["completed_before_interruption"]) == set(jobs)
    analysis = {"manifest_sha256": result["manifest_sha256"], "results_sha256": sha((OUT / "results.json").read_bytes()),
                "groups": groups, "maximum_vocabulary_word_letters": maximum_word,
                "cap32_absolute_letter_bound": 32 * maximum_word,
                "bound_timing": "noticed after execution; not preregistered",
                "interpretation": "cap32 cannot reach 800 letters; this is a feasibility control, not a method-performance baseline; no readability evaluation"}
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
