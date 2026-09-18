"""Audit a controlled POS-pruning result directory without rerunning search."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.pairs import acceptable_pair
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.validator import normalize
from paper.evidence_paths import evidence_path


def audit(run_dir: Path) -> dict:
    provenance = json.loads((run_dir / "provenance.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in
            (run_dir / "trials.jsonl").read_text().splitlines()]
    payload = json.loads(gzip.decompress(
        evidence_path("inputs/brown.json.gz").read_bytes()))
    for name, expected in provenance["inputs_sha256"].items():
        assert hashlib.sha256(evidence_path(name).read_bytes()).hexdigest() == expected
    for name, expected in provenance["source"]["files_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    requested = provenance["configuration"]["vocab"]
    raw = evidence_path("inputs/vocab30k.txt").read_text().split()[:requested]
    vocabulary = {word for word in raw if word in payload["table"]}
    plan = SentencePlan(payload["table"], payload["shapes"],
                        provenance["configuration"]["min_words"],
                        provenance["configuration"]["max_units"] // 2)

    by_trial: dict[int, dict[str, dict]] = {}
    checked_pairs = 0
    for row in rows:
        by_trial.setdefault(row["seed"], {})[row["arm"]] = row
        pairs = [(item["left"], item["right"])
                 for item in row["accepted_pairs"]]
        assert len(pairs) == len(set(pairs)) == row["accepted"]
        digest = hashlib.sha256(
            "".join(f"{left}\0{right}\n" for left, right in sorted(pairs)).encode()
        ).hexdigest()
        assert digest == row["accepted_sha256"]
        assert row["states_generated"] == row["states_pushed"] + row["state_pruned"]
        assert row["states_popped"] <= provenance["configuration"]["node_budget"]
        assert row["accepted"] <= row["eligible_closures"] <= row["closed_states"]
        for left_text, right_text in pairs:
            left, right = left_text.split(), right_text.split()
            assert set(left + right) <= vocabulary
            assert acceptable_pair(left, right,
                                   min_words=provenance["configuration"]["min_words"])
            assert normalize(left_text) == normalize(right_text)[::-1]
            assert plan.complete(left) and plan.complete(right)
            checked_pairs += 1

    assert len(rows) == 2 * len(summary["seeds"])
    missing = {}
    for seed, arms in by_trial.items():
        assert set(arms) == {"terminal", "incremental"}
        assert arms["terminal"]["opening"] == arms["incremental"]["opening"]
        terminal = {(item["left"], item["right"])
                    for item in arms["terminal"]["accepted_pairs"]}
        incremental = {(item["left"], item["right"])
                       for item in arms["incremental"]["accepted_pairs"]}
        if terminal - incremental:
            missing[seed] = len(terminal - incremental)
    assert not missing

    totals = {
        arm: {
            field: sum(row[field] for row in rows if row["arm"] == arm)
            for field in summary["arms"][arm]["totals"]
        }
        for arm in ("terminal", "incremental")
    }
    assert totals == {arm: summary["arms"][arm]["totals"]
                      for arm in ("terminal", "incremental")}
    return {
        "status": "ok",
        "trial_rows": len(rows),
        "paired_trials": len(by_trial),
        "checked_accepted_pair_rows": checked_pairs,
        "terminal_pairs_missing_from_matched_incremental_arm": 0,
        "input_and_recorded_source_hashes_match": True,
        "totals": totals,
        "artifact_sha256": {
            name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
            for name in ("provenance.json", "summary.json", "trials.jsonl", "RESULTS.md")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
