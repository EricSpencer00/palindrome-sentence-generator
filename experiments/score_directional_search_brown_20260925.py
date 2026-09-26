"""Score v4 outputs with the held-out-document Brown word-bigram diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.score_length_stratified_readability import (  # noqa: E402
    HOLDOUT_SEED,
    SEED,
    SHUFFLES,
    model_from_training_documents,
    score,
    split_fileids,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path, output_path: Path) -> dict:
    import nltk
    from nltk.corpus import brown

    input_data = json.loads(input_path.read_text())
    candidates = [row for row in input_data["items"] if row["kind"] == "candidate"]
    if not candidates:
        raise ValueError("no candidate outputs in the frozen input")
    all_fileids = sorted(brown.fileids())
    train_ids, heldout_ids = split_fileids(all_fileids)
    corpus_hash = hashlib.sha256()
    for fileid in all_fileids:
        corpus_hash.update(fileid.encode("utf-8") + b"\0")
        for sentence in brown.sents(fileid):
            corpus_hash.update("\x1f".join(sentence).encode("utf-8") + b"\n")
    model = model_from_training_documents(brown, train_ids)

    rows = []
    for item in candidates:
        text_hash = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
        if text_hash != item["text_sha256"]:
            raise ValueError(f"frozen candidate hash mismatch: {item['id']}")
        scored = score(model, item["id"], item["text"])
        rows.append({
            **scored,
            "task_id": item["pair_id"],
            "resource_configuration": item["source"]["resource_configuration"],
            "method": item["source"]["method"],
            "seed": item["source"]["seed"],
            "band_id": item["source"]["band_id"],
            "text_sha256": text_hash,
        })

    grouped = {}
    for configuration, method in sorted({
        (row["resource_configuration"], row["method"]) for row in rows
    }):
        group = [row for row in rows if row["resource_configuration"] == configuration
                 and row["method"] == method]
        grouped[f"{configuration}:{method}"] = {
            "candidate_count": len(group),
            "mean_order_gain_nats_per_transition": statistics.fmean(
                row["brown_order_gain_vs_own_shuffle"] for row in group
            ),
            "median_order_gain_nats_per_transition": statistics.median(
                row["brown_order_gain_vs_own_shuffle"] for row in group
            ),
            "positive_order_gain_count": sum(
                row["brown_order_gain_vs_own_shuffle"] > 0 for row in group
            ),
        }

    result = {
        "experiment": "directional-search-brown-diagnostic-20260925",
        "status": "programmatic_local_order_diagnostic_not_human_readability_result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": sha256_file(input_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "method": {
            "metric": "Brown add-0.1 word-bigram mean log probability minus the mean over 32 deterministic same-word shuffles",
            "units": "nats per word transition",
            "tokenizer": "ASCII letter tokens lowercased; apostrophes split",
            "candidate_sentence_segmentation": "split on terminal .?! and score sentence boundaries",
            "shuffle": "shuffle pooled item tokens and repartition to original per-sentence token lengths",
            "training_partition": "deterministic 80/20 Brown document split before model construction",
            "split_seed": HOLDOUT_SEED,
            "training_file_count": len(train_ids),
            "heldout_file_count": len(heldout_ids),
            "training_fileid_manifest_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
            "heldout_fileid_manifest_sha256": hashlib.sha256("\n".join(heldout_ids).encode()).hexdigest(),
            "brown_sentences": len(brown.sents()),
            "brown_tokenized_sentence_stream_sha256": corpus_hash.hexdigest(),
            "nltk_version": nltk.__version__,
            "shuffle_count": SHUFFLES,
            "shuffle_seed": SEED,
        },
        "summary_by_configuration_and_method": grouped,
        "scores": rows,
        "limit": "This measures local word-order fit under one corpus bigram model; it does not establish grammar, meaning, coherence, reader preference, or readability.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({
        "output": str(args.output),
        "candidates": len(result["scores"]),
        "groups": result["summary_by_configuration_and_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
