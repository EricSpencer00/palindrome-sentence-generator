from __future__ import annotations

import hashlib
import json
import random

from llm_palindrome.admission import mechanical_admission_checks

from experiments.freshness_register_reader_packet_20261002 import (
    INTACT_CONTROLS,
    SHUFFLED_CONTROLS,
    audit,
    words,
)


EXPERIMENT_ID = "freshness-register-reader-packet-v2-20261002"
SEED = 20261003
TARGETS = (
    {
        "source_id": "target-42",
        "text": "No trace. Note: Spot spoons; snoop; stop. Set one carton.",
        "shuffle": "Set spoons. One note: stop; spot; snoop. No trace carton.",
    },
    {
        "source_id": "target-44",
        "text": "No trace. Note sleet. Spoons snoop. Steel? Set one carton.",
        "shuffle": "Steel spoons. One note: set; snoop; sleet. No trace carton.",
    },
)


def build_payloads() -> tuple[dict, dict]:
    source_rows: list[tuple[str, str, str, str]] = []
    for target in TARGETS:
        source_id = target["source_id"]
        shuffle_id = f"{source_id}-shuffle"
        source_rows.extend(
            [
                (source_id, "candidate", target["text"], shuffle_id),
                (shuffle_id, "shuffled", target["shuffle"], source_id),
            ]
        )
    for index, (intact, shuffled) in enumerate(zip(INTACT_CONTROLS, SHUFFLED_CONTROLS), 1):
        source_rows.extend(
            [
                (f"intact-{index}", "intact_prose", intact, f"intact-{index}-shuffle"),
                (f"intact-{index}-shuffle", "shuffled", shuffled, f"intact-{index}"),
            ]
        )

    rng = random.Random(SEED)
    rng.shuffle(source_rows)
    blind_items = []
    key_items = []
    for order, (source_id, condition, text, pair_id) in enumerate(source_rows, 1):
        opaque_id = hashlib.sha256(f"{SEED}:{source_id}".encode()).hexdigest()[:10]
        blind_items.append({"item_id": opaque_id, "order": order, "text": text})
        key_items.append(
            {
                "item_id": opaque_id,
                "source_id": source_id,
                "condition": condition,
                "matched_pair_source_id": pair_id,
                "word_count": len(words(text)),
            }
        )

    rater = {
        "packet_id": EXPERIMENT_ID,
        "instructions": (
            "Read each item in order without trying to identify how it was made. "
            "Rate the text itself. Do not correct punctuation or silently add words."
        ),
        "questions": [
            {"id": "english", "prompt": "How much does this read as English?", "scale": [1, 7]},
            {"id": "coherence", "prompt": "How coherent is the described scene or instruction?", "scale": [1, 7]},
            {"id": "understanding", "prompt": "How confident are you that you understand it?", "scale": [1, 7]},
            {"id": "complete", "prompt": "Does this feel complete rather than fragmentary?", "scale": [1, 7]},
            {"id": "paraphrase", "prompt": "Briefly paraphrase what it says.", "response": "free_text"},
        ],
        "items": blind_items,
        "readability_claim": "pending independent human ratings",
    }

    target_rows = []
    for target in TARGETS:
        row_audit = audit(target["text"])
        target_rows.append(
            {
                "source_id": target["source_id"],
                "rendered": target["text"],
                "audit": row_audit,
                "mechanical_checks": mechanical_admission_checks(
                    target["text"], min_letters=39, max_letters=100
                ),
                "reader_certified": False,
            }
        )
    key = {
        "packet_id": EXPERIMENT_ID,
        "random_seed": SEED,
        "items": key_items,
        "targets": target_rows,
        "design": {
            "target_count": len(TARGETS),
            "matched_target_shuffle_count": len(TARGETS),
            "intact_prose_control_count": len(INTACT_CONTROLS),
            "matched_control_shuffle_count": len(SHUFFLED_CONTROLS),
            "condition_hidden_from_rater": True,
            "order_randomized": True,
            "fragment_completeness_question_included": True,
            "programmatic_scores_certify_readability": False,
        },
    }
    return rater, key


def main() -> None:
    rater, key = build_payloads()
    print(json.dumps({"rater": rater, "answer_key": key}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
