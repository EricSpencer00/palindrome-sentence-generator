"""Build and summarize a reproducible, model-based language-quality audit.

This is a static proxy audit of the 11 selected texts in paper/week_results.json.
It is deliberately not described as a human readability measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
DEFAULT_INPUT = ROOT / "runs/static-language-verifier-20260925/input.json"
DEFAULT_GPT2 = ROOT / "runs/static-language-verifier-20260925/gpt2.json"
DEFAULT_DEEPSEEK = ROOT / "runs/static-language-verifier-20260925/deepseek.json"
DEFAULT_REPORT = ROOT / "runs/static-language-verifier-20260925/report.json"
SHUFFLE_SEED = 20260925
SHUFFLES_PER_ITEM = 32


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def shuffle_surface(text: str, seed: int) -> str:
    """Shuffle word identities while preserving punctuation and whitespace."""
    import random

    matches = list(WORD_RE.finditer(text))
    tokens = [match.group(0) for match in matches]
    random.Random(seed).shuffle(tokens)
    out = text
    for match, token in reversed(list(zip(matches, tokens))):
        out = out[:match.start()] + token + out[match.end():]
    if Counter(words(out)) != Counter(words(text)):
        raise AssertionError("shuffle changed the word multiset")
    if re.sub(WORD_RE, "<WORD>", out) != re.sub(WORD_RE, "<WORD>", text):
        raise AssertionError("shuffle changed punctuation or whitespace")
    return out


def stable_shuffle_seed(item_id: str, replicate: int) -> int:
    material = f"{SHUFFLE_SEED}:{item_id}:{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def prepare(manifest_path: Path, calibration_path: Path) -> dict:
    """Freeze candidates, their exact Brown matches, and deterministic controls."""
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    candidates = manifest.get("results")
    controls = calibration.get("matched_heldout_prose_controls")
    if not isinstance(candidates, list) or len(candidates) != 11:
        raise ValueError("expected the frozen 11-row paper manifest")
    if not isinstance(controls, list) or len(controls) != len(candidates):
        raise ValueError("expected one held-out Brown control per candidate")

    try:
        from nltk.corpus import brown
    except ImportError as exc:
        raise RuntimeError("Install the pinned NLTK package to materialize Brown controls") from exc

    candidate_by_id = {row["id"]: row for row in candidates}
    control_by_id = {row["matched_candidate_id"]: row for row in controls}
    if set(candidate_by_id) != set(control_by_id):
        raise ValueError("candidate/control IDs differ")

    items: list[dict] = []
    for candidate in candidates:
        item_id = candidate["id"]
        surface = candidate["surface"]
        items.append({
            "id": f"candidate:{item_id}",
            "kind": "candidate",
            "pair_id": item_id,
            "text": surface,
            "word_count": len(words(surface)),
            "text_sha256": sha256_text(surface),
            "source": {
                "manifest": "paper/week_results.json",
                "snapshot_revision": manifest.get("snapshot"),
                "source_artifact": candidate.get("source"),
                "mechanism": candidate.get("mechanism"),
            },
        })

        control = control_by_id[item_id]
        fileid = control["brown_fileid"]
        start, end = control["sentence_indices_half_open"]
        raw_sentences = brown.sents(fileid)[start:end]
        prose = " ".join(" ".join(sentence) for sentence in raw_sentences)
        prose_hash = sha256_text(prose)
        if prose_hash != control["surface_sha256"]:
            raise ValueError(f"Brown control text hash mismatch for {item_id}")
        items.append({
            "id": f"brown:{item_id}",
            "kind": "heldout_brown_prose",
            "pair_id": item_id,
            "text": prose,
            "word_count": len(words(prose)),
            "text_sha256": prose_hash,
            "source": {
                "corpus": "Brown Corpus",
                "nltk_fileid": fileid,
                "sentence_indices_half_open": [start, end],
                "control_manifest": "runs/readability-length-stratified-sentence-aware-20260925.json",
                "matched_candidate_word_count": len(words(surface)),
                "source_word_count": control["source_word_count"],
            },
        })

        for parent_kind, parent_text in (("candidate", surface), ("brown", prose)):
            parent_id = f"{parent_kind}:{item_id}"
            for replicate in range(SHUFFLES_PER_ITEM):
                seed = stable_shuffle_seed(parent_id, replicate)
                shuffled = shuffle_surface(parent_text, seed)
                items.append({
                    "id": f"shuffle:{parent_id}:{replicate:02d}",
                    "kind": f"{parent_kind}_same_word_shuffle",
                    "pair_id": item_id,
                    "parent_id": parent_id,
                    "replicate": replicate,
                    "seed": seed,
                    "text": shuffled,
                    "word_count": len(words(shuffled)),
                    "text_sha256": sha256_text(shuffled),
                })

    result = {
        "experiment": "static-language-verifier-20260925",
        "protocol_version": 1,
        "status": "frozen_inputs_model_outputs_are_not_human_ratings",
        "sources": {
            "candidate_manifest": {
                "path": "paper/week_results.json",
                "sha256": sha256_file(manifest_path),
                "snapshot_revision": manifest.get("snapshot"),
            },
            "brown_control_manifest": {
                "path": "runs/readability-length-stratified-sentence-aware-20260925.json",
                "sha256": sha256_file(calibration_path),
            },
            "corpus": "NLTK Brown corpus; exact file IDs, sentence offsets, and passage hashes are stored per control",
        },
        "method": {
            "candidate_count": len(candidates),
            "heldout_brown_controls_per_candidate": 1,
            "same_word_shuffles_per_candidate_and_brown_control": SHUFFLES_PER_ITEM,
            "shuffle_seed_base": SHUFFLE_SEED,
            "shuffle_seed_derivation": "first 8 SHA-256 bytes of '<base>:<parent-id>:<replicate>', big-endian",
            "shuffle_preserves": "word multiset, punctuation, whitespace, and each word's original casing",
            "candidate_controls_are_selected": "named retrospective exemplars from the paper; not a random sample",
        },
        "items": items,
    }
    return result


def _summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(input_data: dict, gpt2: dict, deepseek: dict) -> dict:
    """Join model outputs to the frozen passages and produce paired results."""
    item_by_id = {item["id"]: item for item in input_data["items"]}
    gpt2_rows = gpt2.get("scores", [])
    gpt2_by_id = {row["id"]: row for row in gpt2_rows}
    if set(gpt2_by_id) != set(item_by_id):
        raise ValueError("GPT-2 scores do not cover the frozen input manifest exactly")

    candidate_rows = []
    for item in input_data["items"]:
        if item["kind"] != "candidate":
            continue
        item_id = item["pair_id"]
        prose = gpt2_by_id[f"brown:{item_id}"]
        candidate = gpt2_by_id[item["id"]]
        candidate_shuffles = [gpt2_by_id[f"shuffle:candidate:{item_id}:{i:02d}"] for i in range(SHUFFLES_PER_ITEM)]
        prose_shuffles = [gpt2_by_id[f"shuffle:brown:{item_id}:{i:02d}"] for i in range(SHUFFLES_PER_ITEM)]
        candidate_rows.append({
            "id": item_id,
            "candidate": item,
            "gpt2_candidate": candidate,
            "gpt2_matched_brown": prose,
            "candidate_minus_brown_mean_token_logprob": candidate["mean_token_logprob"] - prose["mean_token_logprob"],
            "candidate_minus_own_shuffle_mean_token_logprob": candidate["mean_token_logprob"] - statistics.fmean(row["mean_token_logprob"] for row in candidate_shuffles),
            "brown_minus_own_shuffle_mean_token_logprob": prose["mean_token_logprob"] - statistics.fmean(row["mean_token_logprob"] for row in prose_shuffles),
            "candidate_shuffle_scores": candidate_shuffles,
            "brown_shuffle_scores": prose_shuffles,
        })

    comparisons = deepseek.get("comparisons", [])
    pair_summaries = {}
    for baseline in ("candidate_same_word_shuffle",):
        rows = [row for row in comparisons if row.get("baseline_kind") == baseline]
        counts: Counter = Counter()
        for row in rows:
            choice = row.get("parsed", {}).get("choice")
            if choice == "TIE":
                counts["tie"] += 1
            elif choice in {"A", "B"}:
                counts["candidate" if choice == row["candidate_side"] else "baseline"] += 1
            else:
                counts[None] += 1
        pair_summaries[baseline] = {
            "count": len(rows),
            "candidate_preferred": counts.get("candidate", 0),
            "baseline_preferred": counts.get("baseline", 0),
            "ties": counts.get("tie", 0),
            "unparsed": counts.get(None, 0),
            "raw_responses": rows,
        }

    return {
        "experiment": input_data["experiment"],
        "status": "static_model_diagnostics_not_human_readability_results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest": {
            "source_hashes": input_data["sources"],
            "passage_count": len(input_data["items"]),
            "candidate_count": len(candidate_rows),
        },
        "gpt2": {
            "model": gpt2.get("model"),
            "method": gpt2.get("method"),
            "execution": gpt2.get("execution"),
            "input_manifest_sha256": gpt2.get("input_manifest_sha256"),
            "summary": {
                "candidate_mean_token_logprob": _summary([row["gpt2_candidate"]["mean_token_logprob"] for row in candidate_rows]),
                "matched_brown_mean_token_logprob": _summary([row["gpt2_matched_brown"]["mean_token_logprob"] for row in candidate_rows]),
                "candidate_minus_brown_mean_token_logprob": _summary([row["candidate_minus_brown_mean_token_logprob"] for row in candidate_rows]),
                "candidate_minus_own_shuffle_mean_token_logprob": _summary([row["candidate_minus_own_shuffle_mean_token_logprob"] for row in candidate_rows]),
                "brown_minus_own_shuffle_mean_token_logprob": _summary([row["brown_minus_own_shuffle_mean_token_logprob"] for row in candidate_rows]),
                "candidates_scoring_above_matched_brown": sum(row["candidate_minus_brown_mean_token_logprob"] > 0 for row in candidate_rows),
            },
            "paired_rows": candidate_rows,
            "limitations": [
                "GPT-2 token likelihood measures model fit to a text; it is not a direct measure of reader judgments, coherence, or meaning.",
                "The named candidates are retrospective, related exemplars; these scores do not support population-level or causal claims.",
                "Mean likelihood excludes the first GPT-2 token and scores every remaining token without truncation.",
            ],
        },
        "deepseek": {
            "model": deepseek.get("model"),
            "method": deepseek.get("method"),
            "runner_script_sha256": deepseek.get("runner_script_sha256"),
            "sources": deepseek.get("sources"),
            "usage": deepseek.get("usage"),
            "estimated_cost_usd": deepseek.get("estimated_cost_usd"),
            "excluded_preflight": deepseek.get("excluded_preflight"),
            "total_bedrock_estimated_cost_usd": deepseek.get("total_bedrock_estimated_cost_usd", deepseek.get("estimated_cost_usd")),
            "comparison_summaries": pair_summaries,
            "limitations": [
                "DeepSeek responses are judgments from one prompted language model, not human ratings.",
                "The fixed prompt and settings make requests repeatable; the provider does not expose a seed guaranteeing identical output.",
                "These pairwise judgments evaluate only the frozen strings and are not a trained or validated verifier.",
            ],
        },
        "human_evidence": "No project human reader-study responses are included. The static scores do not fill that evidence gap.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, default=ROOT / "paper/week_results.json")
    prepare_parser.add_argument("--calibration", type=Path, default=ROOT / "runs/readability-length-stratified-sentence-aware-20260925.json")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    summary_parser.add_argument("--gpt2", type=Path, default=DEFAULT_GPT2)
    summary_parser.add_argument("--deepseek", type=Path, default=DEFAULT_DEEPSEEK)
    summary_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.calibration)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "passages": len(result["items"]), "candidates": result["method"]["candidate_count"]}, indent=2))
    else:
        input_data = json.loads(args.input.read_text())
        gpt2 = json.loads(args.gpt2.read_text())
        deepseek = json.loads(args.deepseek.read_text())
        report = summarize(input_data, gpt2, deepseek)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "candidate_count": report["input_manifest"]["candidate_count"], "gpt2_gap": report["gpt2"]["summary"]["candidate_minus_brown_mean_token_logprob"], "deepseek": {k: {m: v for m, v in val.items() if m != "raw_responses"} for k, val in report["deepseek"]["comparison_summaries"].items()}}, indent=2))


if __name__ == "__main__":
    main()
