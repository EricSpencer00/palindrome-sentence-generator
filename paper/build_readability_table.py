"""Build the manuscript's long-output/control table from the frozen audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/readability-length-stratified-20260925.json"
DEFAULT_OUTPUT = ROOT / "paper/readability_table.tex"

LABELS = {
    "498-overhang": "498 overhang",
    "568-pinned": "568 parent",
    "616-nora-aron": "616 clause pair",
    "630-god-dog": "630 clause edit",
    "640-event-chain": "640 event edit",
    "672-reverse-chain": "672 search",
    "686-shell-cycle": "686 shell edit",
    "736-mixed-cycle": "736 mixed edit",
    "752-center-path": "752 lineage end",
}


def build_table(report_path: Path = DEFAULT_INPUT) -> str:
    report = json.loads(report_path.read_text())
    controls = {row["matched_candidate_id"]: row
                for row in report["matched_heldout_prose_controls"]}
    candidates = [row for row in report["candidates"] if row["id"] in LABELS]
    candidates.sort(key=lambda row: row["letters"])
    if {row["id"] for row in candidates} != set(LABELS):
        raise ValueError("calibration is missing a manuscript output")

    output_scores = []
    control_scores = []
    rows = []
    for candidate in candidates:
        if not all(candidate["exactness"]["checks"].values()):
            raise ValueError(f"candidate is not independently exact: {candidate['id']}")
        control = controls[candidate["id"]]
        if control["scorer_tokens"] != candidate["scorer_tokens"]:
            raise ValueError(f"token count mismatch: {candidate['id']}")
        output_score = candidate["brown_order_gain_vs_own_shuffle"]
        control_score = control["brown_order_gain_vs_own_shuffle"]
        gap = control_score - output_score
        output_scores.append(output_score)
        control_scores.append(control_score)
        rows.append(
            f"{LABELS[candidate['id']]} & {candidate['letters']} & "
            f"{candidate['scorer_tokens']} & {output_score:.3f} & "
            f"{control_score:.3f} & {gap:.3f} \\\\\n"
        )

    mean_output = sum(output_scores) / len(output_scores)
    mean_control = sum(control_scores) / len(control_scores)
    rows.append(f"Mean & --- & --- & {mean_output:.3f} & {mean_control:.3f} & "
                f"{mean_control - mean_output:.3f} \\\\\n")
    body = "".join(rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n"
        "\\toprule\n"
        "Selected tape & Letters & Tokens & Output & Intact prose & Gap \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Brown word-bigram order gain in nats per transition. Each "
        "output is paired with a held-out intact prose span matched by scorer-token "
        "count. Gap is intact-prose gain minus output gain. Higher gain means "
        "stronger local-order preference over shuffles, not greater readability. "
        "Alphabetic tokenization splits apostrophized forms.}\n"
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
