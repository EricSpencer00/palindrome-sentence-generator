from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from llm_palindrome.admission import mechanical_admission_checks


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "freshness-register-reader-packet-20261002"
SEED = 20261002

CANDIDATE = "No trace. Note: Spot spoons; snoop; stop. Set one carton."
MATCHED_SHUFFLE = "Set spoons. One note: stop; spot; snoop. No trace carton."

INTACT_CONTROLS = (
    "No tracks. Note: Count boxes; inspect labels; stop. Seal one carton.",
    "At dawn, note the loose latch, inspect the crate, and seal one carton.",
    "The clerk spotted the spoons, checked the label, and stopped the cart.",
)

SHUFFLED_CONTROLS = (
    "Seal tracks. One note: inspect boxes; count; carton. No stop labels.",
    "One dawn, seal the inspect latch, note the crate, and loose at carton.",
    "The spoons checked the clerk, stopped the spotted, and label the cart.",
)


def tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.casefold())


def audit(text: str) -> dict:
    normalized = tape(text)
    first_mismatch = next(
        (i for i in range(len(normalized) // 2) if normalized[i] != normalized[-1 - i]),
        None,
    )
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return {
        "letters": len(normalized),
        "two_pointer_exact": bool(normalized) and first_mismatch is None,
        "first_mismatch": first_mismatch,
        "sha256_forward": digest,
        "sha256_reverse": hashlib.sha256(normalized[::-1].encode()).hexdigest(),
    }


def build_payloads() -> tuple[dict, dict]:
    source_rows = [
        ("target", "candidate", CANDIDATE, "target-shuffle"),
        ("target-shuffle", "shuffled", MATCHED_SHUFFLE, "target"),
    ]
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
            {"id": "paraphrase", "prompt": "Briefly paraphrase what it says.", "response": "free_text"},
        ],
        "items": blind_items,
        "readability_claim": "pending independent human ratings",
    }
    key = {
        "packet_id": EXPERIMENT_ID,
        "random_seed": SEED,
        "items": key_items,
        "target": {
            "rendered": CANDIDATE,
            "audit": audit(CANDIDATE),
            "mechanical_checks": mechanical_admission_checks(CANDIDATE, min_letters=39, max_letters=100),
            "provenance": "freshness-indexed live plural residual with LIFO return stack; punctuation-only readability rendering",
        },
        "design": {
            "target_count": 1,
            "matched_target_shuffle_count": 1,
            "intact_prose_control_count": len(INTACT_CONTROLS),
            "matched_control_shuffle_count": len(SHUFFLED_CONTROLS),
            "condition_hidden_from_rater": True,
            "order_randomized": True,
            "programmatic_scores_certify_readability": False,
        },
    }
    return rater, key


def main() -> None:
    rater, key = build_payloads()
    print(json.dumps({"rater": rater, "answer_key": key}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
