"""Recompute the paper's likelihood-sign cross-tab from saved scores, offline.

Run from any directory with Python 3.10+. No model, corpus, or network is used.
Repeated surfaces retain their record-specific shuffle controls. A distinct-text
table is emitted only if all copies agree on both contrast signs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/directional-search-resource-sensitivity-20260925"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def index(rows: list[dict], key: str) -> dict:
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"duplicate {key}")
    return result


def audit(root: Path = ROOT) -> dict:
    raw = (root / RUN / "static-verifier-input-aws.json").read_bytes()
    source = json.loads(raw)
    items = index(source["items"], "id")
    candidates = {k: v for k, v in items.items() if v["kind"] == "candidate"}
    require(len(candidates) == source["candidate_count"], "candidate count mismatch")
    for item in items.values():
        require(digest(item["text"].encode()) == item["text_sha256"], "input text hash mismatch")

    scores = {}
    for model in ("brown", "gpt2"):
        result = json.loads((root / RUN / f"{model}-v4-aws.json").read_bytes())
        require(result["input_manifest_sha256"] == digest(raw), f"{model} input hash mismatch")
        scores[model] = index(result["scores"], "id")
        expected = candidates if model == "brown" else items
        require(scores[model].keys() == expected.keys(), f"{model} score inventory mismatch")
        for key, row in scores[model].items():
            require(row["text_sha256"] == expected[key]["text_sha256"], f"{model} text hash mismatch")

    shuffles = defaultdict(list)
    for item in items.values():
        if item["kind"] != "candidate":
            shuffles[item["pair_id"]].append(item)
    require(set(shuffles) == {x["pair_id"] for x in candidates.values()}, "shuffle pair mismatch")
    records = Counter()
    signs_by_text = defaultdict(set)
    for key, item in candidates.items():
        controls = shuffles[item["pair_id"]]
        require(len(controls) == source["shuffle_count_per_candidate"] == 32, "shuffle count mismatch")
        tape = "".join(c.lower() for c in item["text"] if c.isascii() and c.isalpha())
        require(bool(tape) and tape == tape[::-1], "candidate is not an exact palindrome")
        brown = scores["brown"][key]
        require(brown["shuffle_count"] == 32, "Brown shuffle count mismatch")
        b = brown["brown_order_gain_vs_own_shuffle"]
        require(math.isclose(b, brown["brown_bigram_logprob"] - brown["mean_own_shuffle_logprob"], abs_tol=1e-12), "Brown contrast mismatch")
        g = scores["gpt2"][key]["mean_token_logprob"] - mean(
            scores["gpt2"][control["id"]]["mean_token_logprob"] for control in controls
        )
        require(math.isfinite(b) and math.isfinite(g), "non-finite contrast")
        cell = f"brown_{'positive' if b > 0 else 'nonpositive'}__gpt2_{'positive' if g > 0 else 'nonpositive'}"
        records[cell] += 1
        signs_by_text[item["text_sha256"]].add(cell)

    require(all(len(v) == 1 for v in signs_by_text.values()), "repeated texts disagree on contrast signs; do not collapse")
    distinct = Counter(next(iter(v)) for v in signs_by_text.values())
    qwen = json.loads((root / "runs/aws-qwen3-32b-readability-audit-20260926.json").read_bytes())
    presentations = index(qwen["presentations"], "presentation_id")
    responses = index(qwen["responses"], "presentation_id")
    require(presentations.keys() == responses.keys(), "Qwen response inventory mismatch")
    qwen_candidates = {}
    control_passes = Counter()
    control_counts = Counter()
    for key, presentation in presentations.items():
        response = responses[key]
        require(presentation["text_sha256"] == response["text_sha256"], "Qwen text hash mismatch")
        rating = response["scores"]
        require(all(type(rating[k]) is int and 0 <= rating[k] <= 3 for k in ("grammaticality", "coherent_meaning")), "invalid Qwen score")
        if presentation.get("repeat_of"):
            continue
        condition = presentation["condition"]
        control_counts[condition] += 1
        control_passes[condition] += min(rating["grammaticality"], rating["coherent_meaning"]) >= 2
        if condition == "candidate":
            sha = presentation["text_sha256"]
            require(sha not in qwen_candidates, "duplicate unrepeated Qwen candidate")
            qwen_candidates[sha] = rating
    require(qwen_candidates.keys() == signs_by_text.keys(), "Qwen candidate set mismatch")
    require(all(r == {"grammaticality": 0, "coherent_meaning": 0} for r in qwen_candidates.values()), "candidate score differs from the reported floor")
    return {
        "selected_records": len(candidates),
        "distinct_texts": len(signs_by_text),
        "record_cross_tab": dict(sorted(records.items())),
        "distinct_text_cross_tab": dict(sorted(distinct.items())),
        "repeated_text_signs_agree": True,
        "qwen3_counts_excluding_repeats": dict(sorted(control_counts.items())),
        "qwen3_joint_passes_excluding_repeats": dict(sorted(control_passes.items())),
        "qwen3_candidates_zero_on_both_dimensions": len(qwen_candidates),
        "interpretation": "Descriptive reanalysis of saved scores; signs are not significance tests or human readability labels.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
