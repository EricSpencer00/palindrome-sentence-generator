"""Build the compact NAACL table from frozen static-verifier outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs/static-language-verifier-20260925/report.json"
OUTPUT = ROOT / "paper/static_verifier_table.tex"
LABELS = {
    "38-control": "38-control",
    "54-np-candidate": "54-NP",
    "498-overhang": "Overhang",
    "568-pinned": "Pinned",
    "616-nora-aron": "Nora-Aron",
    "630-god-dog": "God-Dog",
    "640-event-chain": "Events",
    "672-reverse-chain": "Reverse",
    "686-shell-cycle": "Shell",
    "736-mixed-cycle": "Mixed",
    "752-center-path": "Center",
}


def build_tex(report: dict) -> str:
    gpt2 = report["gpt2"]["paired_rows"]
    if set(row["id"] for row in gpt2) != set(LABELS):
        raise ValueError("expected every frozen candidate row exactly once")
    deepseek = report["deepseek"]["comparison_summaries"]["candidate_same_word_shuffle"]["raw_responses"]
    deepseek_by_id = {row["candidate_id"].split(":", 1)[1]: row for row in deepseek}
    if set(deepseek_by_id) != set(LABELS):
        raise ValueError("expected a completed DeepSeek comparison for every candidate")
    if report["deepseek"]["comparison_summaries"]["candidate_same_word_shuffle"]["unparsed"]:
        raise ValueError("cannot build paper table with unparsed DeepSeek choices")

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Text & GPT-2 & Brown & Shuf. & V3.2 \\",
        r"\midrule",
    ]
    for row in gpt2:
        item_id = row["id"]
        choice = deepseek_by_id[item_id]["parsed"]["choice"]
        side = deepseek_by_id[item_id]["candidate_side"]
        if choice == "TIE":
            preference = "Tie"
        elif choice == side:
            preference = "Candidate"
        else:
            preference = "Shuffle"
        lines.append(
            f"{LABELS[item_id]} & "
            f"{row['gpt2_candidate']['mean_token_logprob']:.2f} & "
            f"{row['candidate_minus_brown_mean_token_logprob']:.2f} & "
            f"{row['candidate_minus_own_shuffle_mean_token_logprob']:.2f} & "
            f"{preference} " + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{GPT-2 is mean next-token log probability in nats per token (less negative is higher). Brown and Shuf. report candidate-minus-baseline differences for its word-count-matched held-out Brown span and the mean of 32 same-word shuffles. Positive differences favor the candidate under GPT-2. V3.2 is DeepSeek's choice in one counterbalanced candidate/shuffle pair. These are static model diagnostics, not human readability ratings.}",
        r"\label{tab:static-language-verifier}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    text = build_tex(report)
    args.output.write_text(text)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
