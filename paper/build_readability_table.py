"""Build the manuscript's local-word-order table from the frozen audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/readability-length-stratified-sentence-aware-20260925.json"
DEFAULT_OUTPUT = ROOT / "paper/readability_table.tex"

LABELS = {
    "38-control": "38 inherited example",
    "54-np-candidate": "54 NP candidate",
    "498-overhang": "498 overhang search",
    "568-pinned": "568 pinned text",
    "616-nora-aron": "616 clause insertion",
    "630-god-dog": "630 paired edit",
    "640-event-chain": "640 event edit",
    "672-reverse-chain": "672 bounded search",
    "686-shell-cycle": "686 shell edit",
    "736-mixed-cycle": "736 mixed edit",
    "752-center-path": "752 center path",
}
LONG_IDS = set(LABELS) - {"38-control", "54-np-candidate"}


def checked_rows(report_path: Path) -> tuple[list[dict], dict[str, dict]]:
    report = json.loads(report_path.read_text())
    method = report["method"]
    if report.get("status") != "programmatic_diagnostic_not_human_readability_result":
        raise ValueError("unexpected score status")
    if (method.get("shuffles_per_item") != 32 or
            method.get("random_seed") != 20260925 or
            method.get("training_file_count") != 401 or
            method.get("heldout_file_count") != 99 or
            method.get("nltk_version") != "3.10.3" or
            method.get("metric_units") != "nats per word transition"):
        raise ValueError("score method differs from the audited manuscript protocol")

    candidates = report["candidates"]
    if {row["id"] for row in candidates} != set(LABELS):
        raise ValueError("calibration must include all eleven manuscript cases")
    if len(report["matched_heldout_prose_controls"]) != len(LABELS):
        raise ValueError("calibration must include one matched prose control per case")
    if len(report["length_stratified_heldout_controls"]) != 108:
        raise ValueError("calibration must include all 108 length-stratified controls")

    controls = {row["matched_candidate_id"]: row
                for row in report["matched_heldout_prose_controls"]}
    if set(controls) != set(LABELS):
        raise ValueError("matched prose controls do not cover all manuscript cases")
    candidates = sorted(candidates, key=lambda row: row["letters"])
    for candidate in candidates:
        if not all(candidate["exactness"]["checks"].values()):
            raise ValueError(f"candidate is not independently exact: {candidate['id']}")
        control = controls[candidate["id"]]
        if control["scorer_tokens"] != candidate["scorer_tokens"]:
            raise ValueError(f"token count mismatch: {candidate['id']}")
    return candidates, controls


def mean(rows: list[dict], field: str) -> float:
    return sum(row[field] for row in rows) / len(rows)


def build_table(report_path: Path = DEFAULT_INPUT) -> str:
    candidates, controls = checked_rows(report_path)
    rows = []
    for candidate in candidates:
        control = controls[candidate["id"]]
        output_score = candidate["brown_order_gain_vs_own_shuffle"]
        control_score = control["brown_order_gain_vs_own_shuffle"]
        rows.append(
            f"{LABELS[candidate['id']]} & {candidate['letters']} & "
            f"{candidate['scorer_tokens']} & {output_score:.3f} & "
            f"{control_score:.3f} & {control_score - output_score:.3f} \\\\\n"
        )

    long_candidates = [row for row in candidates if row["id"] in LONG_IDS]
    long_controls = [controls[row["id"]] for row in long_candidates]
    rows.extend((
        "Long set mean ($n=9$) & --- & --- & "
        f"{mean(long_candidates, 'brown_order_gain_vs_own_shuffle'):.3f} & "
        f"{mean(long_controls, 'brown_order_gain_vs_own_shuffle'):.3f} & "
        f"{mean(long_controls, 'brown_order_gain_vs_own_shuffle') - mean(long_candidates, 'brown_order_gain_vs_own_shuffle'):.3f} \\\\\n",
        "All cases mean ($n=11$) & --- & --- & "
        f"{mean(candidates, 'brown_order_gain_vs_own_shuffle'):.3f} & "
        f"{mean(list(controls.values()), 'brown_order_gain_vs_own_shuffle'):.3f} & "
        f"{mean(list(controls.values()), 'brown_order_gain_vs_own_shuffle') - mean(candidates, 'brown_order_gain_vs_own_shuffle'):.3f} \\\\\n",
    ))
    body = "".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n"
        "\\toprule\n"
        "Text & Letters & Tokens & Output & Brown prose & Gap \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Brown word-bigram order gain in nats per transition: observed "
        "score minus the mean of 32 same-word shuffles, preserving sentence "
        "lengths. Each output is paired with a non-overlapping held-out Brown "
        "span matched by scorer-token count. Gap is prose gain minus output gain. "
        "The 38-letter example is inherited; its origin and authorship are not "
        "established by this audit. Higher values indicate stronger local-order "
        "preference under this model, not greater human readability.}\n"
        "\\label{tab:order-gain}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(build_table(args.input))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
