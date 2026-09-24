"""Train and use a DPO pairwise readability *judge* from blinded reader choices.

This is not an exactness verifier and cannot certify readability. DPO needs
chosen/rejected responses; without independent reader choices this script will
not fabricate labels or call an AI proxy human feedback. Its output is a
pairwise ranking signal that must itself be evaluated against held-out readers.

Reader response format (one JSON file)::

    {"raters": [{"rater_id": "R001", "answers": [
      {"task_id": "pair-000", "prefer": "A"}
    ]}]}

The packet is the blinded JSON produced by ``reader_package_v4_20260919.py``.
The answer key is not used to create prompts or labels. Folds are assigned by
task id, so multiple raters of the same pair cannot leak across train/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RUBRIC = (
    "Choose the passage that reads more like ordinary connected English. "
    "Consider grammatical completeness, coherent meaning, and ease of "
    "understanding. Ignore length, punctuation style, and how either passage "
    "was made. Reply with only A or B."
)
READINESS_MIN_TRAIN_TASKS = 20
READINESS_MIN_VALIDATION_TASKS = 8
READINESS_MIN_RATERS = 3


def _prompt(a: str, b: str) -> str:
    return f"{RUBRIC}\n\nPassage A:\n{a}\n\nPassage B:\n{b}\n\nChoice:"


def _fold(task_id: str, seed: int, validation_percent: int) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "validation" if bucket < validation_percent else "train"


def _task_folds(tasks: dict[str, tuple[str, str]], seed: int,
                validation_percent: int) -> tuple[dict[str, str], list[list[str]]]:
    """Keep pairs connected by a repeated passage in the same held-out fold."""
    parent = {task_id: task_id for task_id in tasks}

    def find(task_id: str) -> str:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    passage_owner: dict[str, str] = {}
    for task_id, passages in tasks.items():
        for passage in passages:
            signature = " ".join(passage.casefold().split())
            owner = passage_owner.setdefault(signature, task_id)
            union(task_id, owner)

    components: dict[str, list[str]] = {}
    for task_id in tasks:
        components.setdefault(find(task_id), []).append(task_id)
    groups = sorted((sorted(group) for group in components.values()), key=lambda row: row[0])
    task_folds = {}
    for group in groups:
        # The component key, not an individual response, controls assignment.
        fold = _fold(group[0], seed, validation_percent)
        task_folds.update({task_id: fold for task_id in group})
    return task_folds, groups


def prepare(packet: dict[str, Any], responses: dict[str, Any], *, seed: int = 20260924,
            validation_percent: int = 20) -> dict[str, Any]:
    """Convert blinded A/B reader choices into held-out DPO examples."""
    if not 1 <= validation_percent <= 50:
        raise ValueError("validation_percent must be between 1 and 50")
    items = packet.get("rater_form", {}).get("items")
    if not isinstance(items, list):
        raise ValueError("packet is missing rater_form.items")

    tasks: dict[str, tuple[str, str]] = {}
    for item in items:
        task_id = item.get("task_id")
        a, b = item.get("a", {}).get("text"), item.get("b", {}).get("text")
        if not isinstance(task_id, str) or not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"malformed blinded task: {item!r}")
        if task_id in tasks:
            raise ValueError(f"duplicate task id in packet: {task_id}")
        if not a.strip() or not b.strip() or a == b:
            raise ValueError(f"task {task_id} must have two distinct nonempty passages")
        tasks[task_id] = (a, b)

    raters = responses.get("raters")
    if not isinstance(raters, list):
        raise ValueError("responses must contain a raters list")

    task_folds, connected_groups = _task_folds(tasks, seed, validation_percent)
    seen: set[tuple[str, str]] = set()
    examples: list[dict[str, str]] = []
    scored_task_ids: set[str] = set()
    scored_raters: set[str] = set()
    skipped: dict[str, int] = {"tie_or_unsure": 0, "missing": 0}
    for rater in raters:
        rater_id = rater.get("rater_id")
        answers = rater.get("answers")
        if not isinstance(rater_id, str) or not rater_id.strip() or not isinstance(answers, list):
            raise ValueError(f"malformed rater response: {rater!r}")
        for answer in answers:
            task_id = answer.get("task_id")
            choice = str(answer.get("prefer", "")).strip().upper()
            if task_id not in tasks:
                raise ValueError(f"rater {rater_id} answered unknown task: {task_id!r}")
            key = (rater_id, task_id)
            if key in seen:
                raise ValueError(f"duplicate response for rater/task: {key!r}")
            seen.add(key)
            if choice in ("", "SKIP", "MISSING"):
                skipped["missing"] += 1
                continue
            if choice in ("TIE", "UNSURE", "EQUAL"):
                skipped["tie_or_unsure"] += 1
                continue
            if choice not in ("A", "B"):
                raise ValueError(f"invalid preference {choice!r} for {key!r}")
            a, b = tasks[task_id]
            chosen, rejected = ("A", "B") if choice == "A" else ("B", "A")
            examples.append({"prompt": _prompt(a, b), "chosen": chosen,
                             "rejected": rejected, "task_id": task_id,
                             "rater_id": rater_id,
                             "fold": task_folds[task_id]})
            scored_task_ids.add(task_id)
            scored_raters.add(rater_id)

    train = [row for row in examples if row["fold"] == "train"]
    validation = [row for row in examples if row["fold"] == "validation"]
    train_tasks = {row["task_id"] for row in train}
    validation_tasks = {row["task_id"] for row in validation}
    train_groups = sum(any(task_id in train_tasks for task_id in group)
                       for group in connected_groups)
    validation_groups = sum(any(task_id in validation_tasks for task_id in group)
                           for group in connected_groups)
    # DPOTrainer should see only prompt/completion fields; keep IDs in a
    # separate manifest so no condition, candidate source, or rater leaks in.
    dpo_train = [{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in train]
    dpo_validation = [{k: row[k] for k in ("prompt", "chosen", "rejected")} for row in validation]
    ready = (train_groups >= READINESS_MIN_TRAIN_TASKS
             and validation_groups >= READINESS_MIN_VALIDATION_TASKS
             and len(scored_raters) >= READINESS_MIN_RATERS)
    return {
        "format": "palindrome-readability-dpo-v1",
        "status": "ready_for_dpo_training" if ready else "insufficient_reader_preferences",
        "readiness_thresholds": {
            "train_task_ids": READINESS_MIN_TRAIN_TASKS,
            "validation_task_ids": READINESS_MIN_VALIDATION_TASKS,
            "independent_raters": READINESS_MIN_RATERS,
        },
        "counts": {
            "packet_tasks": len(tasks),
            "preference_examples": len(examples),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "train_task_ids": len(train_tasks),
            "validation_task_ids": len(validation_tasks),
            "train_independent_groups": train_groups,
            "validation_independent_groups": validation_groups,
            "independent_raters": len(scored_raters),
            "skipped": skipped,
        },
        "training_data": dpo_train,
        "validation_data": dpo_validation,
        "split_manifest": {
            "seed": seed,
            "validation_percent": validation_percent,
            "grouping": "connected components of tasks sharing an exact passage; all responses to a task share its fold",
            "task_folds": {task_id: task_folds[task_id]
                           for task_id in sorted(scored_task_ids)},
            "training_task_ids": sorted(train_tasks),
            "validation_task_ids": sorted(validation_tasks),
        },
        "limits": [
            "This is a pairwise preference judge, not an exactness or readability certificate.",
            "AI-generated labels are not accepted by this importer as human responses.",
            "A held-out human comparison evaluation is required before using the judge for ranking.",
        ],
    }


def train(dataset: dict[str, Any], *, model_id: str, output_dir: Path,
          allow_small_pilot: bool = False, epochs: float = 1.0,
          batch_size: int = 2, learning_rate: float = 5e-6) -> None:
    """Fine-tune an A/B preference policy with TRL DPOTrainer.

    Training is intentionally opt-in: this downloads/loads a model and should
    run on the configured remote compute host, not the coordination Mac.
    """
    if dataset.get("format") != "palindrome-readability-dpo-v1":
        raise ValueError("unrecognized prepared dataset format")
    if not dataset.get("training_data") or not dataset.get("validation_data"):
        raise ValueError("DPO needs nonempty chosen/rejected train and validation data")
    if dataset.get("status") != "ready_for_dpo_training" and not allow_small_pilot:
        raise ValueError("reader labels do not meet the training readiness thresholds; "
                         "use --allow-small-pilot only for a clearly labeled pilot")

    try:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("DPO training needs torch, transformers, datasets, and trl installed "
                           "on the selected compute host") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        beta=0.1,
        max_length=1024,
        max_prompt_length=960,
    )
    trainer = DPOTrainer(
        model=model_id,
        args=config,
        train_dataset=Dataset.from_list(dataset["training_data"]),
        eval_dataset=Dataset.from_list(dataset["validation_data"]),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    (output_dir / "readability-judge-manifest.json").write_text(
        json.dumps({
            "status": "dpo_pairwise_judge_not_readability_certificate",
            "base_model": model_id,
            "dataset_counts": dataset["counts"],
            "training_status": dataset["status"],
            "heldout_human_evaluation_required": True,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def compare(model_id: str, passage_a: str, passage_b: str) -> dict[str, str]:
    """Ask a trained pairwise judge for a blind A/B preference (triage only)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    prompt = _prompt(passage_a, passage_b)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=3, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    match = re.match(r"^\s*([AB])(?:\b|\W)", raw, re.IGNORECASE)
    return {"preference": match.group(1).upper() if match else "unparsed",
            "raw": raw,
            "status": "uncalibrated_pairwise_triage_only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="turn blinded reader choices into DPO data")
    prepare_parser.add_argument("--packet", type=Path, required=True)
    prepare_parser.add_argument("--responses", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260924)
    prepare_parser.add_argument("--validation-percent", type=int, default=20)
    train_parser = sub.add_parser("train", help="train the pairwise judge with DPO")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--out-dir", type=Path, required=True)
    train_parser.add_argument("--allow-small-pilot", action="store_true")
    train_parser.add_argument("--epochs", type=float, default=1.0)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=5e-6)
    compare_parser = sub.add_parser("compare", help="compare two passages with a trained judge")
    compare_parser.add_argument("--model", required=True)
    compare_parser.add_argument("--a", required=True)
    compare_parser.add_argument("--b", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare(json.loads(args.packet.read_text(encoding="utf-8")),
                         json.loads(args.responses.read_text(encoding="utf-8")),
                         seed=args.seed, validation_percent=args.validation_percent)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "output": str(args.out)}, indent=2))
    elif args.command == "train":
        train(json.loads(args.dataset.read_text(encoding="utf-8")), model_id=args.model,
              output_dir=args.out_dir, allow_small_pilot=args.allow_small_pilot,
              epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=args.learning_rate)
    else:
        print(json.dumps(compare(args.model, args.a, args.b), indent=2))


if __name__ == "__main__":
    main()
