"""Preflight and reference implementation for a live GPT-2 character decoder.

The proposed state is deliberately different from post-hoc reranking: lexical
actions are expanded one character at a time, and a local GPT-2 next-token
log-probability is charged before an action survives the seam frontier.  This
module does not run that decoder when the novelty registry reports an overlap;
the current registry already retains GPT-2 bilateral prefix/token-lattice and
reverse-segmentation routes.  The preflight therefore records a reproducible
pivot without loading a model or counting duplicate evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "live-gpt2-character-decoder-preflight-20260916"
SIGNATURE = (
    "live-gpt2-next-token-character-decoder|"
    "bilateral-normal-order-lexeme-expansion|semantic-valency-state|"
    "character-seam-obligation|independent-tape-hash-admission|"
    "mismatch-heldout-repair"
)
EVIDENCE = ROOT / "runs" / "live-gpt2-character-decoder-preflight-20260916.json"
MIN_LETTERS = 39
MAX_LETTERS = 220


def tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_validation(rendered: str) -> dict[str, object]:
    """Recompute exactness and hashes without consulting decoder state."""
    normalized = tape(rendered)
    mismatches = [
        {"index": i, "left": normalized[i], "right": normalized[-1 - i]}
        for i in range(len(normalized) // 2)
        if normalized[i] != normalized[-1 - i]
    ]
    return {
        "letters": len(normalized),
        "normalized_tape": normalized,
        "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "exact": bool(normalized) and not mismatches,
        "two_pointer": {"exact": bool(normalized) and not mismatches, "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None},
    }


@dataclass(frozen=True)
class Lexeme:
    word: str
    pos: str
    sense: str


@dataclass(frozen=True)
class LiveState:
    """A live normal-order state; no finished tape is reverse-decoded."""

    left_text: str
    right_text: str
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    left_tape: str
    right_tape: str
    lm_logprob: float
    semantic_roles: tuple[str, ...]


def next_token_logprob(model, tokenizer, prefix: str, candidate: str) -> float:
    """Score an action using GPT-2 logits before committing its characters."""
    import torch

    context_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
    action_ids = tokenizer(" " + candidate.strip(), add_special_tokens=False, return_tensors="pt").input_ids
    if context_ids.shape[1] == 0 or action_ids.shape[1] == 0:
        return float("-inf")
    ids = torch.cat((context_ids, action_ids), dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0]
    logprobs = logits[:-1].log_softmax(-1)
    targets = ids[0, 1:]
    picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
    start = context_ids.shape[1] - 1
    return float(picked[start:].sum().item())


def live_character_expand(
    states: Sequence[LiveState],
    left_options: Iterable[Lexeme],
    right_options: Iterable[Lexeme],
    model,
    tokenizer,
    *,
    beam: int = 32,
    seam_check: Callable[[str, str], bool] | None = None,
) -> list[LiveState]:
    """Expand lexical actions while charging GPT-2 and checking the seam live.

    The caller supplies a seam obligation for the current slot boundary.  A
    production run would use a length-bounded frontier obligation; this small
    primitive is intentionally model-agnostic for deterministic unit tests.
    """
    seam_check = seam_check or (lambda _left, _right: True)
    expanded: list[LiveState] = []
    for state in states:
        for left, right in ((l, r) for l in left_options for r in right_options):
            left_text = (state.left_text + " " + left.word).strip()
            right_text = (state.right_text + " " + right.word).strip()
            left_tape, right_tape = tape(left_text), tape(right_text)
            if not seam_check(left_tape, right_tape):
                continue
            score = state.lm_logprob + next_token_logprob(model, tokenizer, state.left_text, left.word)
            score += next_token_logprob(model, tokenizer, state.right_text, right.word)
            expanded.append(LiveState(left_text, right_text, state.left_words + (left.word,), state.right_words + (right.word,), left_tape, right_tape, score, state.semantic_roles + (left.sense, right.sense)))
    return sorted(expanded, key=lambda s: (-s.lm_logprob, s.left_text, s.right_text))[:beam]


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries = registry.get("entries", [])
    excluded = registry.get("excluded", [])
    prior = [row for row in entries + excluded if row.get("id") != EXPERIMENT_ID]
    current = set(re.findall(r"[a-z0-9]+", SIGNATURE.lower()))
    common = {"a", "an", "and", "audit", "character", "complete", "exact", "independent", "normal", "order", "the", "two", "with"}
    current -= common
    overlaps: list[dict[str, object]] = []
    for row in prior:
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "").lower())) - common
        shared = current & other
        union = current | other
        score = len(shared) / len(union) if union else 0.0
        # GPT-2 plus live bilateral character/token expansion is a retained
        # construction dimension even when generic signature Jaccard is low.
        gpt2_family = "gpt2" in other and bool({"character", "token", "prefix", "decoder"} & other)
        if score >= 0.25 or gpt2_family:
            overlaps.append({"id": row.get("id"), "jaccard": round(score, 6), "shared_atoms": sorted(shared), "reason": "retained GPT-2 character/token/prefix decoder family" if gpt2_family else "signature overlap"})
    # Keep the most directly relevant prior routes first.
    overlaps.sort(key=lambda row: ("gpt2" not in str(row["id"]), -row["jaccard"], str(row["id"])))
    return {"registry_entries": len(entries), "excluded_routes": len(excluded), "overlaps": overlaps, "blocked": bool(overlaps), "performed_before_model_load": True}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    payload: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "novelty_preflight": preflight,
        "method": "Live character-synchronous lexical expansion charges local GPT-2 next-token log-probabilities before each lexical action and carries semantic role state through a bilateral seam frontier; completed text is independently audited afterward.",
        "config": {"model": "gpt2-local-cache", "decoder": "character-synchronous-next-token", "normal_order_lexeme_actions": True, "posthoc_reranking": False, "half_tape_decoder": False, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS},
        "stats": {"model_loaded": 0, "live_expansions": 0, "rendered_probes": 0, "exact": 0, "mechanically_admitted": 0, "reader_eligible": 0},
        "rendered_candidates": [],
        "rendered_probes": [],
        "repair": {"operator": "At the first live seam mismatch, replace only the held-out lexeme with the same POS/sense and re-expand its characters under GPT-2 next-token probability; rerun independent tape/hash validation and the shared mechanical gate.", "status": "not_run"},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "model_loaded": False, "source_sentences_copied": False, "known_palindromes_used": False, "readability_certificate": False},
        "pivot": {"status": "preflight_blocked", "reason": "The registry retains GPT-2 bilateral prefix/token-lattice and reverse-character decoder families; running this lane would duplicate the same model-in-the-loop character decoder dimension.", "overlaps": preflight["overlaps"], "next_route": "Use a non-neural semantic construction or obtain an explicitly new model/state dimension before loading GPT-2."},
    }
    if not preflight["blocked"]:
        raise AssertionError("unexpectedly unblocked: this lane must not silently run without a registry decision")
    return payload


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": payload["pivot"]["status"], "overlaps": len(payload["pivot"]["overlaps"])}, sort_keys=True))


if __name__ == "__main__":
    main()
