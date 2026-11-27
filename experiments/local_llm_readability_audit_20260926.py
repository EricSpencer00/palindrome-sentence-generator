#!/usr/bin/env python3
"""Run a blinded local-model audit of the frozen v4 sentence set.

The experiment keeps exact letter-palindrome verification deterministic and
separate from the model's rubric scores.  The model sees one text per request;
it does not see condition labels, source IDs, or the paired item.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from static_language_verifier_20260925 import (  # noqa: E402
    sha256_text,
    shuffle_surface,
    stable_shuffle_seed,
    words,
)


RUN_ID = "local-llm-readability-audit-20260926"
MODEL = "qwen2.5:3b"
ORDER_SEED = 20260926
SHUFFLE_REPLICATES = (0, 1, 2)
POSITIVE_CONTROL_COUNT = 12
REPEAT_PAIR_COUNT = 6
INPUT_CANDIDATES = ROOT / "paper/human_eval_v4_packet/sealed_key.json"
INPUT_STATIC = ROOT / (
    "runs/directional-search-resource-sensitivity-20260925/"
    "static-verifier-input-aws.json"
)
INPUT_LENGTH_CONTROLS = ROOT / "runs/readability-length-stratified-20260925.json"
DEFAULT_OUTPUT = ROOT / "runs/local-llm-readability-audit-20260926.json"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

SYSTEM_PROMPT = """You are a blinded evaluator of one short English text.
Judge only the exact text provided. Do not add words, repair it, reorder it,
guess the author's intent, or reward formal patterns, repetition, rhyme, or
symmetry. Score grammaticality and recoverable meaning independently. Ordinary
words placed in a sequence without a stable sentence meaning should receive low
scores even if each word is familiar. Do not provide chain-of-thought."""

USER_PROMPT_TEMPLATE = """Rate this text on two independent integer scales from 0 to 3.

grammaticality:
0 = no recoverable sentence parse / word salad;
1 = isolated grammatical phrases, but major sentence-level breaks;
2 = a sentence parse is mostly recoverable, though strained or awkward;
3 = natural, grammatical connected English.

coherent_meaning:
0 = no recoverable proposition or event;
1 = a few word associations, but their relations are unclear;
2 = one plausible interpretation is recoverable, though incomplete or vague;
3 = a clear meaning with understandable relations between its parts.

Do not explain or quote the text. Return only the requested JSON.

TEXT:
{text}"""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grammaticality": {"type": "integer", "minimum": 0, "maximum": 3},
        "coherent_meaning": {"type": "integer", "minimum": 0, "maximum": 3},
    },
    "required": ["grammaticality", "coherent_meaning"],
    "additionalProperties": False,
}

DECODE = {
    "temperature": 0,
    "top_p": 1,
    "seed": ORDER_SEED,
    "num_predict": 128,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json_with_hash(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def opaque_id(prefix: str, material: str) -> str:
    return f"{prefix}-{sha256_bytes(material.encode('utf-8'))[:12].upper()}"


def normalized_case(text: str) -> str:
    """Lowercase letters and capitalize only the first alphabetic character."""
    value = text.lower()
    match = re.search(r"[a-z]", value)
    if match is None:
        return value
    return value[:match.start()] + value[match.start()].upper() + value[match.end():]


def letter_tape(text: str) -> str:
    return "".join(ch.lower() for ch in text if "a" <= ch.lower() <= "z")


def word_palindrome(text: str) -> bool:
    tokens = [token.casefold() for token in WORD_RE.findall(text)]
    return bool(tokens) and tokens == tokens[::-1]


def load_candidate_rows(
    packet_path: Path, static_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packet, packet_hash = read_json_with_hash(packet_path)
    static, static_hash = read_json_with_hash(static_path)
    if packet.get("source_input_sha256") != static_hash:
        raise ValueError("review packet does not match the frozen static-verifier input")
    if static.get("status") != "frozen_confirmatory_outputs_and_deterministic_controls":
        raise ValueError("static-verifier source is not the frozen v4 input")
    if static.get("candidate_count") != 66 or static.get("shuffle_count_per_candidate") != 32:
        raise ValueError("unexpected candidate or shuffle counts in static input")

    by_id = {row["id"]: row for row in static["items"]}
    source_shuffles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in static["items"]:
        if row.get("kind") == "candidate_same_word_shuffle":
            source_shuffles[row.get("source", {}).get("candidate_id", "")].append(row)

    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for candidate in packet.get("candidates", []):
        text = candidate.get("text")
        text_hash = sha256_text(text) if isinstance(text, str) else ""
        if not text or text_hash != candidate.get("text_sha256"):
            raise ValueError(f"candidate hash mismatch: {candidate.get('item_id')!r}")
        if text_hash in seen_hashes:
            raise ValueError("packet contains duplicate candidate text")
        seen_hashes.add(text_hash)
        tape = letter_tape(text)
        if not tape or tape != tape[::-1]:
            raise ValueError(f"candidate fails independent letter-tape check: {candidate['item_id']}")

        origins = candidate.get("source_rows", [])
        if not origins:
            raise ValueError(f"candidate has no frozen source rows: {candidate['item_id']}")
        origin = min(origins, key=lambda row: row["source_item_id"])
        source_id = origin["source_item_id"]
        source_row = by_id.get(source_id)
        if not source_row or source_row.get("kind") != "candidate":
            raise ValueError(f"source candidate missing: {source_id}")
        if source_row.get("text") != text or source_row.get("text_sha256") != text_hash:
            raise ValueError(f"source candidate content mismatch: {source_id}")

        all_shuffles = source_shuffles.get(source_id, [])
        if len(all_shuffles) != 32:
            raise ValueError(f"expected 32 archived shuffles for {source_id}")
        by_replicate = {row.get("source", {}).get("replicate"): row for row in all_shuffles}
        if set(by_replicate) != set(range(32)):
            raise ValueError(f"shuffle replicates are incomplete for {source_id}")

        selected.append({
            "item_id": candidate["item_id"],
            "text": text,
            "text_sha256": text_hash,
            "normalized_text": normalized_case(text),
            "letter_palindrome": True,
            "word_order_palindrome": word_palindrome(text),
            "band_id_for_calibration_only": int(origin["band_id"]),
            "source_item_id_private": source_id,
            "shuffle_rows": by_replicate,
        })

    if len(selected) != 52:
        raise ValueError(f"expected 52 unique packet candidates, found {len(selected)}")
    return selected, {
        "candidate_packet_sha256": packet_hash,
        "static_verifier_input_sha256": static_hash,
    }


def load_brown_controls(
    controls_path: Path, candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    controls_doc, controls_hash = read_json_with_hash(controls_path)
    controls = controls_doc.get("length_stratified_heldout_controls")
    if not isinstance(controls, list) or len(controls) != 108:
        raise ValueError("expected the frozen 108-item held-out Brown control manifest")
    try:
        import nltk
        from nltk.corpus import brown
    except ImportError as exc:
        raise RuntimeError("NLTK and the Brown Corpus are required to reconstruct positive controls") from exc

    targets_by_band: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        targets_by_band[row["band_id_for_calibration_only"]].append(row)
    if any(len(targets_by_band[band]) < 4 for band in (1, 2, 3)):
        raise ValueError("need at least four unique candidate targets in each length band")

    selected_targets: list[dict[str, Any]] = []
    for band in (1, 2, 3):
        ranked = sorted(
            targets_by_band[band],
            key=lambda row: sha256_bytes(f"positive-target:{ORDER_SEED}:{row['item_id']}".encode()),
        )
        selected_targets.extend(ranked[:4])

    remaining = list(controls)
    positive_rows: list[dict[str, Any]] = []
    for target in selected_targets:
        selected_control = min(
            remaining,
            key=lambda row: (
                abs(int(row["source_word_count"]) - len(words(target["text"]))),
                sha256_bytes(
                    f"positive-control:{ORDER_SEED}:{target['item_id']}:{row['id']}".encode()
                ),
            ),
        )
        remaining.remove(selected_control)
        fileid = selected_control["brown_fileid"]
        start, end = selected_control["sentence_indices_half_open"]
        raw_sentences = brown.sents(fileid)[start:end]
        surface = " ".join(" ".join(sentence) for sentence in raw_sentences)
        if sha256_text(surface) != selected_control["surface_sha256"]:
            raise ValueError(f"Brown sentence hash mismatch for {selected_control['id']}")
        positive_rows.append({
            "item_id": opaque_id("B", selected_control["id"]),
            "target_item_id_private": target["item_id"],
            "source_manifest_id_private": selected_control["id"],
            "text": surface,
            "normalized_text": normalized_case(surface),
            "text_sha256": sha256_text(surface),
            "source_word_count": len(words(surface)),
            "target_word_count": len(words(target["text"])),
            "brown_fileid": fileid,
            "sentence_indices_half_open": [start, end],
        })

    source_method = controls_doc.get("method", {})
    return positive_rows, controls_hash, {
        "corpus": "Brown Corpus held-out files",
        "nltk_version": nltk.__version__,
        "heldout_file_count": source_method.get("heldout_file_count"),
        "heldout_fileid_manifest_sha256": source_method.get("heldout_fileid_manifest_sha256"),
        "prose_control_manifest_sha256": controls_hash,
        "positive_control_selection": (
            "four candidate-matched targets per length band; nearest source word count; "
            "deterministic SHA-256 tie break; no replacement; selected without model scores"
        ),
    }


def make_presentations(
    candidates: list[dict[str, Any]],
    brown_controls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = []
    for candidate in candidates:
        pair_id = candidate["item_id"]
        base.append({
            "condition": "candidate",
            "pair_id": pair_id,
            "item_id": pair_id,
            "text": candidate["normalized_text"],
            "source_text_sha256": candidate["text_sha256"],
            "letter_palindrome": candidate["letter_palindrome"],
            "word_order_palindrome": candidate["word_order_palindrome"],
        })
        for replicate in SHUFFLE_REPLICATES:
            shuffle = candidate["shuffle_rows"][replicate]
            source = shuffle.get("source", {})
            raw_text = shuffle["text"]
            if source.get("preserves") != "word multiset, punctuation, whitespace, and each word's original casing":
                raise ValueError(f"unexpected control contract for {shuffle['id']}")
            if Counter(w.casefold() for w in words(raw_text)) != Counter(
                w.casefold() for w in words(candidate["text"])
            ):
                raise ValueError(f"word multiset mismatch for {shuffle['id']}")
            if WORD_RE.sub("<WORD>", raw_text) != WORD_RE.sub("<WORD>", candidate["text"]):
                raise ValueError(f"punctuation/spacing mismatch for {shuffle['id']}")
            base.append({
                "condition": "candidate_same_word_shuffle",
                "pair_id": pair_id,
                "item_id": opaque_id("S", f"{pair_id}:{replicate}:{shuffle['text_sha256']}"),
                "text": normalized_case(raw_text),
                "source_text_sha256": shuffle["text_sha256"],
                "shuffle_replicate": replicate,
                "same_word_multiset": True,
                "letter_palindrome": letter_tape(raw_text) == letter_tape(raw_text)[::-1],
            })

    for control in brown_controls:
        item_id = control["item_id"]
        parent_id = f"brown-positive:{control['source_manifest_id_private']}"
        seed = stable_shuffle_seed(parent_id, 0)
        shuffled = shuffle_surface(control["text"], seed)
        if Counter(w.casefold() for w in words(shuffled)) != Counter(
            w.casefold() for w in words(control["text"])
        ):
            raise ValueError(f"Brown shuffle word multiset mismatch for {item_id}")
        if WORD_RE.sub("<WORD>", shuffled) != WORD_RE.sub("<WORD>", control["text"]):
            raise ValueError(f"Brown shuffle punctuation mismatch for {item_id}")
        base.extend([
            {
                "condition": "heldout_brown_prose",
                "pair_id": item_id,
                "item_id": item_id,
                "text": control["normalized_text"],
                "source_text_sha256": control["text_sha256"],
                "source_word_count": control["source_word_count"],
                "letter_palindrome": letter_tape(control["text"]) == letter_tape(control["text"])[::-1],
                "brown_source_id": control["source_manifest_id_private"],
                "brown_fileid": control["brown_fileid"],
                "brown_sentence_indices_half_open": control["sentence_indices_half_open"],
            },
            {
                "condition": "heldout_brown_same_word_shuffle",
                "pair_id": item_id,
                "item_id": opaque_id("T", f"{item_id}:00:{sha256_text(shuffled)}"),
                "text": normalized_case(shuffled),
                "source_text_sha256": sha256_text(shuffled),
                "shuffle_replicate": 0,
                "shuffle_seed": seed,
                "same_word_multiset": True,
                "letter_palindrome": letter_tape(shuffled) == letter_tape(shuffled)[::-1],
                "brown_source_id": control["source_manifest_id_private"],
                "brown_fileid": control["brown_fileid"],
                "brown_sentence_indices_half_open": control["sentence_indices_half_open"],
            },
        ])

    # Select two candidate pairs in each target-length band for hidden repeats.
    # The model receives no condition, item ID, or repeat marker.
    repeated_pairs: list[str] = []
    for band in (1, 2, 3):
        band_rows = [row for row in candidates if row["band_id_for_calibration_only"] == band]
        ordered = sorted(
            band_rows,
            key=lambda row: sha256_bytes(f"repeat-pair:{ORDER_SEED}:{row['item_id']}".encode()),
        )
        repeated_pairs.extend(row["item_id"] for row in ordered[:2])
    if len(repeated_pairs) != REPEAT_PAIR_COUNT:
        raise ValueError("expected exactly six repeat pairs")

    original_records: list[dict[str, Any]] = []
    for item in base:
        presentation_id = opaque_id(
            "V", f"{ORDER_SEED}:{item['condition']}:{item['item_id']}:{item.get('shuffle_replicate', 'base')}"
        )
        original_records.append({**item, "presentation_id": presentation_id, "repeat_of": None})
    repeats: list[dict[str, Any]] = []
    for pair_id in repeated_pairs:
        to_repeat = [
            row for row in original_records
            if row["pair_id"] == pair_id
            and row["condition"] in {"candidate", "candidate_same_word_shuffle"}
            and row.get("shuffle_replicate", 0) == 0
        ]
        if len(to_repeat) != 2:
            raise ValueError(f"could not identify candidate + first shuffle for repeat pair {pair_id}")
        for original in to_repeat:
            duplicate_id = opaque_id("V", f"{ORDER_SEED}:repeat:{original['presentation_id']}")
            repeats.append({**original, "presentation_id": duplicate_id,
                            "repeat_of": original["presentation_id"]})
    if len(repeats) != 12:
        raise ValueError("expected 12 hidden repeat presentations")

    presentations = original_records + repeats
    presentations.sort(
        key=lambda row: sha256_bytes(f"{ORDER_SEED}:order:{row['presentation_id']}".encode())
    )
    for ordinal, row in enumerate(presentations, start=1):
        row["presentation_order"] = ordinal
        row["text_sha256"] = sha256_text(row["text"])
    if len({row["presentation_id"] for row in presentations}) != len(presentations):
        raise ValueError("presentation ID collision")
    return presentations


def get_json(url: str, timeout: float = 10) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def model_facts(base_url: str, model: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    tags = get_json(base + "/api/tags")
    matches = [row for row in tags.get("models", []) if row.get("name") == model]
    if len(matches) != 1:
        raise ValueError(f"expected one installed {model!r} model; found {len(matches)}")
    try:
        version = get_json(base + "/api/version").get("version")
    except (HTTPError, URLError, TimeoutError, ValueError):
        version = None
    match = matches[0]
    return {
        "model": model,
        "model_digest": match.get("digest"),
        "model_size_bytes": match.get("size"),
        "model_details": match.get("details", {}),
        "ollama_version": version,
    }


def ask_model(base_url: str, model: str, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base = base_url.rstrip("/")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": DECODE,
        "keep_alive": "5m",
    }
    request = Request(
        base + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    raw = result.get("message", {}).get("content", "")
    parsed = json.loads(raw)
    scores = {}
    for key in ("grammaticality", "coherent_meaning"):
        value = parsed.get(key)
        if type(value) is not int or not 0 <= value <= 3:
            raise ValueError(f"invalid {key} value from model: {value!r}")
        scores[key] = value
    return {"scores": scores, "raw_json": raw}, {
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
        "load_duration_ns": result.get("load_duration"),
        "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
        "eval_duration_ns": result.get("eval_duration"),
    }


def atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def public_presentation(row: dict[str, Any]) -> dict[str, Any]:
    """Keep hashes and source locators, not copied Brown Corpus sentences."""
    return {
        key: value for key, value in row.items()
        if key != "text" and not key.endswith("_private")
    }


def compute_analysis(presentations: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["presentation_id"]: row for row in presentations}
    response_by_id = {row["presentation_id"]: row for row in responses}
    primary: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    repeats: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in responses:
        presentation = by_id[row["presentation_id"]]
        if presentation["repeat_of"]:
            repeats.append((presentation, row))
        elif presentation["condition"] in {"candidate", "candidate_same_word_shuffle"}:
            pair = primary[presentation["pair_id"]]
            condition = presentation["condition"]
            if condition == "candidate":
                pair[condition] = row
            else:
                pair.setdefault(condition, {})[presentation["shuffle_replicate"]] = row

    candidate_rows = [row for row in primary.values() if row.get("candidate")]
    if len(candidate_rows) != 52 or any(
        len(row.get("candidate_same_word_shuffle", {})) != 3 for row in candidate_rows
    ):
        raise ValueError("cannot analyze incomplete candidate/shuffle pairs")

    def joint_pass(result: dict[str, Any]) -> int:
        scores = result["scores"]
        return int(scores["grammaticality"] >= 2 and scores["coherent_meaning"] >= 2)

    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        score_keys = ("grammaticality", "coherent_meaning")
        return {
            "n": len(rows),
            "joint_pass_at_least_2_on_both": sum(joint_pass(row) for row in rows),
            "joint_pass_rate": sum(joint_pass(row) for row in rows) / len(rows) if rows else None,
            "score_means": {
                key: round(statistics.mean(row["scores"][key] for row in rows), 4)
                for key in score_keys
            } if rows else {},
            "score_medians": {
                key: statistics.median(row["scores"][key] for row in rows)
                for key in score_keys
            } if rows else {},
            "score_histograms": {
                key: {str(score): sum(row["scores"][key] == score for row in rows)
                      for score in range(4)}
                for key in score_keys
            } if rows else {},
        }

    candidate_scores = [row["candidate"] for row in candidate_rows]
    shuffled_scores = [
        row["candidate_same_word_shuffle"][replicate]
        for row in candidate_rows for replicate in sorted(row["candidate_same_word_shuffle"])
    ]
    pair_passes = []
    dimension_comparisons: dict[str, dict[str, int]] = {
        key: {"candidate_higher": 0, "tie": 0, "candidate_lower": 0}
        for key in ("grammaticality", "coherent_meaning")
    }
    for pair in candidate_rows:
        candidate = pair["candidate"]
        shuffles = [pair["candidate_same_word_shuffle"][r] for r in sorted(pair["candidate_same_word_shuffle"])]
        candidate_pass = joint_pass(candidate)
        shuffle_pass_rate = statistics.mean(joint_pass(row) for row in shuffles)
        pair_passes.append({
            "candidate_joint_pass": candidate_pass,
            "shuffle_joint_pass_fraction": shuffle_pass_rate,
        })
        for key in dimension_comparisons:
            candidate_value = candidate["scores"][key]
            shuffle_mean = statistics.mean(row["scores"][key] for row in shuffles)
            if candidate_value > shuffle_mean:
                dimension_comparisons[key]["candidate_higher"] += 1
            elif candidate_value < shuffle_mean:
                dimension_comparisons[key]["candidate_lower"] += 1
            else:
                dimension_comparisons[key]["tie"] += 1

    candidate_pass_count = sum(row["candidate_joint_pass"] for row in pair_passes)
    shuffle_mean_pass_rate = statistics.mean(
        row["shuffle_joint_pass_fraction"] for row in pair_passes
    )
    candidate_vs_shuffle_table = {
        "candidate_pass_and_shuffle_majority_pass": 0,
        "candidate_pass_and_shuffle_majority_fail": 0,
        "candidate_fail_and_shuffle_majority_pass": 0,
        "candidate_fail_and_shuffle_majority_fail": 0,
    }
    for pair in candidate_rows:
        candidate_pass = joint_pass(pair["candidate"])
        shuffle_passes = sum(
            joint_pass(pair["candidate_same_word_shuffle"][r]) for r in range(3)
        )
        if candidate_pass and shuffle_passes >= 2:
            key = "candidate_pass_and_shuffle_majority_pass"
        elif candidate_pass:
            key = "candidate_pass_and_shuffle_majority_fail"
        elif shuffle_passes >= 2:
            key = "candidate_fail_and_shuffle_majority_pass"
        else:
            key = "candidate_fail_and_shuffle_majority_fail"
        candidate_vs_shuffle_table[key] += 1

    brown_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for response in responses:
        presentation = by_id[response["presentation_id"]]
        if presentation["repeat_of"] is None and presentation["condition"] in {
            "heldout_brown_prose", "heldout_brown_same_word_shuffle"
        }:
            brown_by_pair[presentation["pair_id"]][presentation["condition"]] = response

    duplicate_exact = 0
    duplicate_scores = []
    for presentation, response in repeats:
        original = response_by_id.get(presentation["repeat_of"])
        if original is None:
            raise ValueError("repeat presentation has no original response")
        same = original["scores"] == response["scores"]
        duplicate_exact += int(same)
        duplicate_scores.append({
            "same_scores": same,
            "original_scores": original["scores"],
            "repeat_scores": response["scores"],
        })

    brown_passes = sum(
        joint_pass(values["heldout_brown_prose"]) for values in brown_by_pair.values()
    )
    brown_shuffle_passes = sum(
        joint_pass(values["heldout_brown_same_word_shuffle"]) for values in brown_by_pair.values()
    )
    calibration_pass = brown_passes >= 8 and brown_shuffle_passes <= 3
    return {
        "candidate_unique_texts": summarize_rows(candidate_scores),
        "candidate_matched_shuffles_3_per_text": summarize_rows(shuffled_scores),
        "heldout_brown_prose_positive_controls": summarize_rows([
            values["heldout_brown_prose"] for values in brown_by_pair.values()
        ]),
        "heldout_brown_matched_shuffles": summarize_rows([
            values["heldout_brown_same_word_shuffle"] for values in brown_by_pair.values()
        ]),
        "primary_finite_set_comparison": {
            "outcome": "candidate has both grammaticality >= 2 and coherent meaning >= 2",
            "n_unique_candidate_texts": len(candidate_rows),
            "candidate_joint_passes": candidate_pass_count,
            "candidate_joint_pass_rate": candidate_pass_count / len(candidate_rows),
            "mean_of_three_shuffle_joint_pass_rates": round(shuffle_mean_pass_rate, 4),
            "paired_difference_candidate_minus_mean_shuffle_rate": round(
                candidate_pass_count / len(candidate_rows) - shuffle_mean_pass_rate, 4
            ),
            "candidate_vs_shuffle_transition_counts": candidate_vs_shuffle_table,
            "dimension_comparison_counts_vs_three_shuffle_mean": dimension_comparisons,
            "interpretation": (
                "descriptive for these frozen, success-conditioned texts and these deterministic controls; "
                "not a human readability estimate or a population-level inference"
            ),
        },
        "positive_control_separation": {
            "brown_prose_joint_passes": brown_passes,
            "brown_shuffle_joint_passes": brown_shuffle_passes,
            "calibration_gate_passes": calibration_pass,
            "predeclared_interpretation_gate": (
                "if fewer than 8/12 Brown prose controls pass both >=2, or more than 3/12 Brown shuffles do, "
                "treat the candidate score distribution as inconclusive under this judge"
            ),
        },
        "hidden_repeat_agreement": {
            "repeat_pairs": len(repeats),
            "exact_two_score_agreements": duplicate_exact,
            "exact_score_agreement_rate": duplicate_exact / len(repeats) if repeats else None,
            "comparisons": duplicate_scores,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435",
                        help="Ollama API base URL, normally an SSH-forwarded remote endpoint")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate inputs and print counts without contacting the model")
    parser.add_argument("--resume", action="store_true",
                        help="resume a partial output after verifying its run fingerprint")
    args = parser.parse_args()

    candidates, input_hashes = load_candidate_rows(INPUT_CANDIDATES, INPUT_STATIC)
    brown_controls, length_hash, brown_source = load_brown_controls(
        INPUT_LENGTH_CONTROLS, candidates
    )
    presentations = make_presentations(candidates, brown_controls)
    if args.prepare_only:
        public_manifest = [public_presentation(row) for row in presentations]
        if any("text" in row or any(key.endswith("_private") for key in row)
               for row in public_manifest):
            raise ValueError("persisted presentation manifest would expose text or private mapping keys")
        print(json.dumps({
            "candidates": len(candidates),
            "candidate_shuffles": len(candidates) * len(SHUFFLE_REPLICATES),
            "brown_positive_controls": len(brown_controls),
            "brown_shuffles": len(brown_controls),
            "hidden_repeats": sum(row["repeat_of"] is not None for row in presentations),
            "total_presentations": len(presentations),
            "all_candidates_letter_palindromes": all(row["letter_palindrome"] for row in candidates),
            "word_order_palindrome_count": sum(row["word_order_palindrome"] for row in candidates),
            "candidate_shuffle_word_multiset_valid": True,
            "brown_control_source_manifest_sha256": length_hash,
            "inputs": input_hashes,
            "brown_source": brown_source,
        }, indent=2))
        return

    facts = model_facts(args.base_url, args.model)
    script_hash = sha256_bytes(Path(__file__).read_bytes())
    prompt_sha = sha256_bytes((SYSTEM_PROMPT + "\0" + USER_PROMPT_TEMPLATE).encode("utf-8"))
    presentation_manifest = [public_presentation(row) for row in presentations]
    run_fingerprint = sha256_bytes(json.dumps({
        "inputs": input_hashes,
        "length_control_manifest_sha256": length_hash,
        "brown_source": brown_source,
        "model": facts,
        "scorer_script_sha256": script_hash,
        "prompt_sha256": prompt_sha,
        "decode": DECODE,
        "output_schema": OUTPUT_SCHEMA,
        "replicates": SHUFFLE_REPLICATES,
        "order_seed": ORDER_SEED,
        "presentation_manifest": presentation_manifest,
    }, sort_keys=True).encode("utf-8"))

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"{args.output} exists; select a new --output or use --resume")
    if args.resume:
        if not args.output.exists():
            raise FileNotFoundError("--resume requested but output file does not exist")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("run_fingerprint") != run_fingerprint:
            raise ValueError("partial output fingerprint differs from current inputs/model/prompt")
        responses = result.get("responses", [])
    else:
        result = {
            "experiment_id": RUN_ID,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": run_fingerprint,
            "source_artifacts": {
                "candidate_packet": "paper/human_eval_v4_packet/sealed_key.json",
                "static_verifier_input": INPUT_STATIC.relative_to(ROOT).as_posix(),
                "heldout_brown_control_manifest": INPUT_LENGTH_CONTROLS.relative_to(ROOT).as_posix(),
                "scorer_script": Path(__file__).relative_to(ROOT).as_posix(),
                "scorer_script_sha256": script_hash,
                **input_hashes,
                "heldout_brown_control_manifest_sha256": length_hash,
            },
            "model_facts": facts,
            "method": {
                "model_blind_to_condition": True,
                "one_text_per_request": True,
                "condition_labels_and_ids_in_prompt": False,
                "candidate_text_count": 52,
                "candidate_shuffle_replicates": list(SHUFFLE_REPLICATES),
                "brown_positive_controls": len(brown_controls),
                "brown_shuffle_controls": len(brown_controls),
                "brown_source": brown_source,
                "candidate_and_control_case_policy": (
                    "lowercase all letters, capitalize only the first alphabetic character; "
                    "preserve punctuation and whitespace"
                ),
                "shuffle_policy": (
                    "three frozen controls per unique candidate, selected from archived replicates 00, 01, 02; "
                    "each preserves the source word multiset, punctuation, and whitespace"
                ),
                "rubric": {
                    "grammaticality": {
                        "0": "no recoverable sentence parse / word salad",
                        "1": "isolated grammatical phrases with major sentence-level breaks",
                        "2": "mostly recoverable parse, strained or awkward",
                        "3": "natural, grammatical connected English",
                    },
                    "coherent_meaning": {
                        "0": "no recoverable proposition or event",
                        "1": "word associations without clear relations",
                        "2": "one plausible but incomplete or vague interpretation",
                        "3": "clear meaning with understandable relations",
                    },
                    "rationale": "not requested; no quoted or generated explanations",
                },
                "primary_outcome": "joint score >=2 on both dimensions; candidate minus mean of its three matched shuffles",
                "positive_control_gate": "inconclusive if <8/12 prose pass both >=2 or >3/12 prose shuffles pass both >=2",
                "hidden_repeat_count": 12,
                "decoding": DECODE,
                "prompt_sha256": prompt_sha,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": USER_PROMPT_TEMPLATE,
                "output_schema": OUTPUT_SCHEMA,
                "ollama_chat_structured_output": True,
                "external_hosted_model_calls": False,
                "aws_calls": False,
                "remote_ollama_requests_via_ssh_forward": True,
            },
            "presentations": presentation_manifest,
            "responses": [],
        }
        responses = []

    completed = {row["presentation_id"] for row in responses}
    result["status"] = "running"
    result.pop("last_error", None)
    atomic_write(args.output, result)
    started = time.monotonic()
    try:
        for index, presentation in enumerate(presentations, start=1):
            presentation_id = presentation["presentation_id"]
            if presentation_id in completed:
                continue
            parsed, api_metrics = ask_model(args.base_url, args.model, presentation["text"])
            row = {
                "presentation_id": presentation_id,
                "scores": parsed["scores"],
                "raw_json": parsed["raw_json"],
                "api_metrics": api_metrics,
                "request_completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            responses.append(row)
            completed.add(presentation_id)
            result["responses"] = responses
            atomic_write(args.output, result)
            print(f"scored {index}/{len(presentations)}", flush=True)
        result["analysis"] = compute_analysis(presentations, responses)
        result["status"] = "complete"
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["elapsed_seconds_this_turn"] = round(time.monotonic() - started, 3)
        atomic_write(args.output, result)
    except Exception as exc:
        result["responses"] = responses
        result["status"] = "failed_partial"
        result["last_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds_this_turn"] = round(time.monotonic() - started, 3)
        atomic_write(args.output, result)
        raise

    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
