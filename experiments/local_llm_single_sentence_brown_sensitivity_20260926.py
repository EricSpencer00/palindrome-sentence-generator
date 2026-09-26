#!/usr/bin/env python3
"""Post-hoc Qwen calibration sensitivity using individual held-out Brown sentences.

This separate diagnostic cannot retroactively pass the original audit's
calibration gate or validate the original candidate scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import local_llm_readability_audit_20260926 as base  # noqa: E402


RUN_ID = "local-llm-single-sentence-brown-sensitivity-20260926"
SELECTION_SEED = 20260927
PRESENTATION_SEED = 20260928
CONTROL_COUNT = 12
DEFAULT_OUTPUT = ROOT / "runs/local-llm-single-sentence-brown-sensitivity-20260926.json"
TERMINAL_RE = re.compile(r"[.!?][\"'”’)]*\s*$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "text"}


def prepare() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import nltk
    from nltk.corpus import brown

    candidates, candidate_hashes = base.load_candidate_rows(
        base.INPUT_CANDIDATES, base.INPUT_STATIC
    )
    controls_doc, length_hash = base.read_json_with_hash(base.INPUT_LENGTH_CONTROLS)
    method = controls_doc.get("method", {})

    holdout_seed = "brown-readability-file-holdout-20260925"
    all_fileids = sorted(brown.fileids())
    heldout_fileids = [
        fileid for fileid in all_fileids
        if int(hashlib.sha256(f"{holdout_seed}:{fileid}".encode()).hexdigest()[:8], 16) % 5 == 0
    ]
    heldout_hash = sha256_bytes("\n".join(heldout_fileids).encode())
    if heldout_hash != method.get("heldout_fileid_manifest_sha256"):
        raise ValueError("Brown held-out file split does not match the frozen manifest")

    previously_used: set[tuple[str, int]] = set()
    for group_name in ("matched_heldout_prose_controls", "length_stratified_heldout_controls"):
        for row in controls_doc.get(group_name, []):
            start, end = row["sentence_indices_half_open"]
            previously_used.update((row["brown_fileid"], index) for index in range(start, end))

    candidate_targets: list[dict[str, Any]] = []
    by_band: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for row in candidates:
        by_band[row["band_id_for_calibration_only"]].append(row)
    for band in (1, 2, 3):
        ranked = sorted(
            by_band[band],
            key=lambda row: base.sha256_bytes(
                f"single-sentence-target:{SELECTION_SEED}:{row['item_id']}".encode()
            ),
        )
        if len(ranked) < 4:
            raise ValueError(f"not enough unique candidates in band {band}")
        candidate_targets.extend(ranked[:4])

    brown_sentences: list[dict[str, Any]] = []
    for fileid in heldout_fileids:
        for index, sentence in enumerate(brown.sents(fileid)):
            if (fileid, index) in previously_used:
                continue
            text = " ".join(sentence).strip()
            count = len(base.words(text))
            if not 12 <= count <= 40 or not TERMINAL_RE.search(text):
                continue
            brown_sentences.append({
                "text": text,
                "word_count": count,
                "brown_fileid": fileid,
                "sentence_index": index,
                "text_sha256": base.sha256_text(text),
            })

    chosen: list[dict[str, Any]] = []
    used_fileids: set[str] = set()
    for target in candidate_targets:
        target_count = len(base.words(target["text"]))
        options = [row for row in brown_sentences if row["brown_fileid"] not in used_fileids]
        if not options:
            raise ValueError("ran out of unused held-out Brown files")
        selected = min(options, key=lambda row: (
            abs(row["word_count"] - target_count),
            base.sha256_bytes(
                f"single-sentence-control:{SELECTION_SEED}:{target['item_id']}:{row['brown_fileid']}:{row['sentence_index']}".encode()
            ),
        ))
        used_fileids.add(selected["brown_fileid"])
        selected = {
            **selected,
            "control_id": base.opaque_id("B", f"{selected['brown_fileid']}:{selected['sentence_index']}:{selected['text_sha256']}"),
            "target_word_count": target_count,
            "target_band": target["band_id_for_calibration_only"],
        }
        chosen.append(selected)

    presentations: list[dict[str, Any]] = []
    for control in chosen:
        positive_id = control["control_id"]
        positive_text = base.normalized_case(control["text"])
        shuffle_seed = base.stable_shuffle_seed(
            f"single-brown:{control['brown_fileid']}:{control['sentence_index']}", 0
        )
        shuffled = base.shuffle_surface(control["text"], shuffle_seed)
        if Counter(word.casefold() for word in base.words(shuffled)) != Counter(
            word.casefold() for word in base.words(control["text"])
        ):
            raise ValueError(f"Brown control shuffle changed words: {positive_id}")
        if base.WORD_RE.sub("<WORD>", shuffled) != base.WORD_RE.sub("<WORD>", control["text"]):
            raise ValueError(f"Brown control shuffle changed punctuation/spacing: {positive_id}")
        presentations.extend([
            {
                "condition": "individual_brown_sentence",
                "pair_id": positive_id,
                "item_id": positive_id,
                "text": positive_text,
                "source_text_sha256": control["text_sha256"],
                "brown_fileid": control["brown_fileid"],
                "sentence_index": control["sentence_index"],
                "sentence_span_length": 1,
                "word_count": control["word_count"],
                "target_word_count": control["target_word_count"],
                "target_band": control["target_band"],
            },
            {
                "condition": "individual_brown_sentence_same_word_shuffle",
                "pair_id": positive_id,
                "item_id": base.opaque_id("S", f"{positive_id}:{shuffle_seed}:{base.sha256_text(shuffled)}"),
                "text": base.normalized_case(shuffled),
                "source_text_sha256": base.sha256_text(shuffled),
                "brown_fileid": control["brown_fileid"],
                "sentence_index": control["sentence_index"],
                "sentence_span_length": 1,
                "word_count": control["word_count"],
                "target_word_count": control["target_word_count"],
                "target_band": control["target_band"],
                "shuffle_seed": shuffle_seed,
                "same_word_multiset": True,
            },
        ])

    presentations.sort(key=lambda row: base.sha256_bytes(
        f"{PRESENTATION_SEED}:{row['condition']}:{row['item_id']}".encode()
    ))
    for order, row in enumerate(presentations, start=1):
        row["presentation_id"] = base.opaque_id(
            "V", f"{PRESENTATION_SEED}:{row['condition']}:{row['item_id']}"
        )
        row["presentation_order"] = order
        row["text_sha256"] = base.sha256_text(row["text"])
    if len(presentations) != 24 or len(chosen) != CONTROL_COUNT:
        raise ValueError("expected 12 individual Brown sentences and 12 matched shuffles")

    source = {
        "corpus": "Brown Corpus held-out document split",
        "nltk_version": nltk.__version__,
        "heldout_seed": holdout_seed,
        "heldout_file_count": len(heldout_fileids),
        "heldout_fileid_manifest_sha256": heldout_hash,
        "length_control_manifest_sha256": length_hash,
        "prior_controls_excluded": True,
        "individual_sentence_selection": (
            "four hash-selected candidate target lengths per band; nearest word count; "
            "deterministic SHA-256 tie break; no file or sentence reuse; no model scores"
        ),
        "candidate_packet_sha256": candidate_hashes["candidate_packet_sha256"],
        "static_verifier_input_sha256": candidate_hashes["static_verifier_input_sha256"],
    }
    return presentations, source


def summarize(presentations: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    by_presentation = {row["presentation_id"]: row for row in presentations}
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    response_by_id = {row["presentation_id"]: row for row in responses}
    for result in responses:
        presentation = by_presentation[result["presentation_id"]]
        by_pair.setdefault(presentation["pair_id"], {})[presentation["condition"]] = result

    def joint(item: dict[str, Any]) -> int:
        score = item["scores"]
        return int(score["grammaticality"] >= 2 and score["coherent_meaning"] >= 2)

    summary: dict[str, Any] = {}
    transitions = {"sentence_and_shuffle_pass": 0, "sentence_only_pass": 0,
                   "shuffle_only_pass": 0, "neither_pass": 0}
    candidate_rows, shuffle_rows = [], []
    for paired in by_pair.values():
        sentence = paired["individual_brown_sentence"]
        shuffle = paired["individual_brown_sentence_same_word_shuffle"]
        candidate_rows.append(sentence)
        shuffle_rows.append(shuffle)
        key = (
            "sentence_and_shuffle_pass" if joint(sentence) and joint(shuffle) else
            "sentence_only_pass" if joint(sentence) else
            "shuffle_only_pass" if joint(shuffle) else "neither_pass"
        )
        transitions[key] += 1

    def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "joint_passes": sum(joint(row) for row in rows),
            "joint_pass_rate": sum(joint(row) for row in rows) / len(rows),
            "score_means": {
                key: round(statistics.mean(row["scores"][key] for row in rows), 4)
                for key in ("grammaticality", "coherent_meaning")
            },
            "score_histograms": {
                key: {str(score): sum(row["scores"][key] == score for row in rows)
                      for score in range(4)}
                for key in ("grammaticality", "coherent_meaning")
            },
        }

    sentence_passes = sum(joint(row) for row in candidate_rows)
    shuffle_passes = sum(joint(row) for row in shuffle_rows)
    summary["posthoc_single_sentence_sensitivity"] = {
        "intact_single_sentence_controls": group_stats(candidate_rows),
        "matched_same_word_shuffles": group_stats(shuffle_rows),
        "paired_transitions": transitions,
        "posthoc_gate_passes": sentence_passes >= 8 and shuffle_passes <= 3,
        "caveat": (
            "separate post-hoc calibration sensitivity only; it cannot change the first run's failed gate "
            "or validate the original candidate scores"
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default=base.MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    presentations, brown_source = prepare()
    public_presentations = [public_row(row) for row in presentations]
    if args.prepare_only:
        print(json.dumps({
            "individual_brown_sentences": 12,
            "matched_shuffles": 12,
            "total_presentations": 24,
            "all_sentence_spans_have_one_sentence": all(
                row["sentence_span_length"] == 1 for row in public_presentations
            ),
            "unique_brown_files": len({row.get("brown_fileid") for row in public_presentations}),
            "brown_source": brown_source,
        }, indent=2))
        return

    facts = base.model_facts(args.base_url, args.model)
    script_hash = sha256_file(Path(__file__))
    prompt_hash = base.sha256_bytes(
        (base.SYSTEM_PROMPT + "\0" + base.USER_PROMPT_TEMPLATE).encode()
    )
    fingerprint = base.sha256_bytes(json.dumps({
        "brown_source": brown_source,
        "model_facts": facts,
        "scorer_script_sha256": script_hash,
        "prompt_sha256": prompt_hash,
        "decode": base.DECODE,
        "presentation_manifest": public_presentations,
    }, sort_keys=True).encode())

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; choose another output or pass --resume")
    if args.resume:
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("run_fingerprint") != fingerprint:
            raise ValueError("resume fingerprint differs from frozen inputs/model/prompt")
        responses = result.get("responses", [])
    else:
        result = {
            "experiment_id": RUN_ID,
            "status": "running",
            "run_fingerprint": fingerprint,
            "source_artifacts": {
                "brown_control_manifest": base.INPUT_LENGTH_CONTROLS.relative_to(ROOT).as_posix(),
                "candidate_packet": base.INPUT_CANDIDATES.relative_to(ROOT).as_posix(),
                "static_verifier_input": base.INPUT_STATIC.relative_to(ROOT).as_posix(),
                "scorer_script": Path(__file__).relative_to(ROOT).as_posix(),
                "scorer_script_sha256": script_hash,
            },
            "model_facts": facts,
            "method": {
                "status": "post-hoc calibration sensitivity; not a replacement for first-run gate",
                "prompt_and_decode": "identical to local-llm-readability-audit-20260926",
                "presentation_order_seed": PRESENTATION_SEED,
                "positive_sentence_count": 12,
                "matched_shuffle_count": 12,
                "individual_brown_sentence_selection": brown_source["individual_sentence_selection"],
                "gate": "at least 8/12 intact sentences pass both >=2 and at most 3/12 shuffles pass both >=2",
                "no_candidate_scores_rerun": True,
                "no_AWS_or_hosted_model_calls": True,
                "brown_source": brown_source,
                "prompt_sha256": prompt_hash,
                "decoding": base.DECODE,
                "system_prompt": base.SYSTEM_PROMPT,
                "user_prompt_template": base.USER_PROMPT_TEMPLATE,
                "output_schema": base.OUTPUT_SCHEMA,
            },
            "presentations": public_presentations,
            "responses": [],
        }
        responses = []

    completed = {row["presentation_id"] for row in responses}
    result["status"] = "running"
    result.pop("last_error", None)
    atomic_write(args.output, result)
    started = time.monotonic()
    try:
        for presentation in presentations:
            presentation_id = presentation["presentation_id"]
            if presentation_id in completed:
                continue
            parsed, api_metrics = base.ask_model(
                args.base_url, args.model, presentation["text"]
            )
            responses.append({
                "presentation_id": presentation_id,
                "scores": parsed["scores"],
                "raw_json": parsed["raw_json"],
                "api_metrics": api_metrics,
                "request_completed_utc": datetime.now(timezone.utc).isoformat(),
            })
            completed.add(presentation_id)
            result["responses"] = responses
            atomic_write(args.output, result)
            print(f"scored {len(completed)}/24", flush=True)
        result["analysis"] = summarize(presentations, responses)
        result["status"] = "complete"
        result["started_utc"] = result.get("started_utc") or datetime.now(timezone.utc).isoformat()
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds_this_turn"] = round(time.monotonic() - started, 3)
        atomic_write(args.output, result)
    except Exception as exc:
        result["responses"] = responses
        result["status"] = "failed_partial"
        result["last_error"] = f"{type(exc).__name__}: {exc}"
        atomic_write(args.output, result)
        raise

    print(json.dumps(result["analysis"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
