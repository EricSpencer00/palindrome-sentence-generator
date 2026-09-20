"""Build a reproducible blinded reader package for the v4 frontier.

The package separates the rater form from its answer key.  Raters see only
randomized A/B passages; the key retains source, condition, and independent
exact audits.  Shuffled controls are diagnostics, never generated outputs.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.v4 import BEST_KNOWN_TEXT, independent_audit
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "reader-package-v4-20260919"
SEED = 20260919

# These are short intact prose controls, authored for calibration rather than
# presented as generated palindromes.  Their labels stay out of the rater form.
INTACT_CONTROLS = (
    "At dawn, a careful scribe reads the harbor letter while Diana watches the tide.",
    "The young herald carries a sonnet to the court, and the players answer with song.",
    "A baker carries a map; an artist answers some bells.",
    "The herald carries the letter through the hall; the actors keep the oath near the grove.",
)

# Fresh exact-by-construction output awaiting human judgment.  The package
# deliberately keeps this as an unlabeled passage in the rater form; exactness
# is disclosed only in the separated answer key.
GENERATED_CANDIDATES = (
    ("semordnilap-noel-scene-56", "semordnilap-poetic-clause-20261001", "No evil Noel deliver desserts raw; war stressed reviled Leon live on."),
)


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def _shuffle(text: str, seed: int) -> str:
    words = _words(text)
    random.Random(seed).shuffle(words)
    punctuation = "."
    if text.rstrip()[-1:] in ".!?":
        punctuation = text.rstrip()[-1]
    return " ".join(words) + punctuation


def _frontier() -> list[dict[str, object]]:
    rows = [
        {"source_id": "best-known-38", "source": "half-tape-grammar-csp-20260919", "text": BEST_KNOWN_TEXT},
    ]
    rows.extend(
        {"source_id": source_id, "source": source, "text": text}
        for source_id, source, text in GENERATED_CANDIDATES
    )
    rows.extend(
        {"source_id": f"intact-control-{i}", "source": "authored-reader-control", "text": text}
        for i, text in enumerate(INTACT_CONTROLS, 1)
    )
    return rows


def build(seed: int = SEED) -> dict[str, object]:
    rater_items = []
    answer_key = []
    for index, row in enumerate(_frontier()):
        intact_id = f"item-{index:03d}-intact"
        shuffled_id = f"item-{index:03d}-shuffled"
        shuffled = _shuffle(str(row["text"]), seed + index)
        pair = [(intact_id, str(row["text"]), "intact"), (shuffled_id, shuffled, "shuffled")]
        order = list(pair)
        random.Random(seed * 100 + index).shuffle(order)
        rater_items.append(
            {
                "task_id": f"pair-{index:03d}",
                "a": {"item_id": order[0][0], "text": order[0][1]},
                "b": {"item_id": order[1][0], "text": order[1][1]},
                "question": "Which passage reads more like intact English? Choose A or B; ignore length and palindrome status.",
            }
        )
        for item_id, rendered, condition in pair:
            answer_key.append(
                {
                    "item_id": item_id,
                    "task_id": f"pair-{index:03d}",
                    "source_id": row["source_id"],
                    "source": row["source"],
                    "condition": condition,
                    "rendered": rendered,
                    "audit": independent_audit(rendered),
                    "mechanical_checks": mechanical_admission_checks(rendered, min_letters=30, max_letters=2000),
                }
            )
    return {
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "status": "blinded_package_ready_human_ratings_pending",
        "rater_form": {"instructions": "Rate only connected English. Ignore length, exactness, and source.", "items": rater_items},
        "answer_key": answer_key,
        "reproducibility": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "randomization": "Python Random with explicit integer seed; answer key is separate from rater form",
            "pair_count": len(rater_items),
        },
        "reader_protocol": {
            "intact_vs_shuffled": True,
            "randomized_blinded_order": True,
            "human_readability_is_required": True,
            "programmatic_metrics_certify_readability": False,
            "next_action": "collect independent ratings and report pairwise preference with rater IDs and exclusions",
        },
    }


if __name__ == "__main__":
    result = build()
    out = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pairs": len(result["rater_form"]["items"]), "seed": result["seed"], "output": str(out)}))
