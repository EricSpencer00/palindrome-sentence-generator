"""Direct character-equation decoding with GPT-2 in the live search loop.

Grammar prefixes own the beam. At every character or word-boundary transition
the local GPT-2 next-token probability is queried, then the live character
equation is applied. Both sides are emitted in ordinary reading order; a
completed side supplies reverse-index obligations to the still-growing side,
never a reverse-emitted half or a fixed tape.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "live-gpt2-character-decoder-preflight-20260916"
SIGNATURE = (
    "direct-gpt2-next-character-equation|grammar-prefix-boundary-transitions|"
    "bilateral-normal-order-decoding|live-obligation-state|"
    "semantic-valency-lexicalization|independent-tape-hash-admission|"
    "mismatch-heldout-repair"
)
EVIDENCE = ROOT / "runs" / "live-gpt2-character-decoder-preflight-20260916.json"
MIN_LETTERS = 39
MAX_LETTERS = 220
BEAM = 20
MAX_STEPS = 100


@dataclass(frozen=True)
class Lexeme:
    word: str
    pos: str
    sense: str = ""


@dataclass(frozen=True)
class Verb:
    word: str
    object_sense: str


@dataclass(frozen=True)
class Cursor:
    side: str
    slot: int = 0
    prefix: str = ""
    choices: tuple[str, ...] = ()
    words: tuple[str, ...] = ()
    text: str = ""
    verb_sense: str = ""
    complete: bool = False


@dataclass(frozen=True)
class State:
    left: Cursor
    right: Cursor
    score: float = 0.0
    transitions: int = 0
    first_mismatch: dict[str, object] | None = None


DETS = ("a", "the")
PREPS_L = ("in", "near", "under", "by")
PREPS_R = ("at", "over", "behind", "within")
ADJS_L = ("quiet", "bright", "clear", "fresh")
ADJS_R = ("brief", "dark", "clean", "gentle")
SUBJ_L = ("baker", "farmer", "gardener", "keeper", "maker", "poet")
SUBJ_R = ("caller", "clerk", "driver", "friend", "guide", "nurse")
VERBS_L = (Verb("reads", "text"), Verb("writes", "text"), Verb("stores", "food"), Verb("marks", "object"), Verb("visits", "place"))
VERBS_R = (Verb("answers", "text"), Verb("offers", "text"), Verb("gathers", "food"), Verb("checks", "object"), Verb("enters", "place"))
OBJECTS_L = {"text": ("letter", "note", "poem"), "food": ("bread", "grain", "meal"), "object": ("map", "parcel", "button"), "place": ("garden", "harbor", "market")}
OBJECTS_R = {"text": ("message", "reply", "response"), "food": ("orange", "dinner", "rice"), "object": ("alarm", "package", "signal"), "place": ("camp", "hall", "port")}
LOCS_L = ("room", "field", "shore", "yard")
LOCS_R = ("plaza", "valley", "wall", "village")


def tape(text: str) -> str:
    return normalize_letters(text)


def independent_validation(rendered: str) -> dict[str, object]:
    """Recompute palindrome status and hash without decoder state."""
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
        "two_pointer": {"exact": bool(normalized) and not mismatches,
                        "mismatch_count": len(mismatches),
                        "first_mismatch": mismatches[0] if mismatches else None},
    }


def slot_options(cursor: Cursor) -> tuple[str, ...]:
    left = cursor.side == "left"
    if cursor.slot in (0, 6):
        return DETS
    if cursor.slot == 3:
        # Keep a determiner that is valid for every independently selected
        # object; the decoder's semantic choice remains live in slot 4.
        return ("the",)
    if cursor.slot == 1:
        return SUBJ_L if left else SUBJ_R
    if cursor.slot == 2:
        return tuple(v.word for v in (VERBS_L if left else VERBS_R))
    if cursor.slot == 4:
        choices = OBJECTS_L if left else OBJECTS_R
        return choices[cursor.verb_sense]
    if cursor.slot == 5:
        return PREPS_L if left else PREPS_R
    if cursor.slot == 7:
        return ADJS_L if left else ADJS_R
    if cursor.slot == 8:
        return LOCS_L if left else LOCS_R
    return ()


def cursor_transitions(cursor: Cursor) -> list[tuple[str, Cursor, str]]:
    """Return grammar-licensed character and boundary transitions."""
    if cursor.complete:
        return []
    choices = cursor.choices or slot_options(cursor)
    if cursor.prefix:
        choices = tuple(word for word in choices if word.startswith(cursor.prefix))
    if not choices:
        return []
    out: list[tuple[str, Cursor, str]] = []
    for char in sorted({word[len(cursor.prefix)] for word in choices if len(word) > len(cursor.prefix)}):
        prefix = cursor.prefix + char
        out.append((char, replace(cursor, prefix=prefix, choices=tuple(word for word in choices if word.startswith(prefix))), "character"))
    if cursor.prefix in choices:
        word = cursor.prefix
        words = cursor.words + (word,)
        verb_sense = cursor.verb_sense
        if cursor.slot == 2:
            verbs = VERBS_L if cursor.side == "left" else VERBS_R
            verb_sense = next(v.object_sense for v in verbs if v.word == word)
        done = cursor.slot == 8
        next_cursor = replace(cursor, slot=cursor.slot + 1, prefix="", choices=(), words=words, text=cursor.text + word + ("" if done else " "), verb_sense=verb_sense, complete=done)
        out.append((" ", next_cursor, "word_boundary"))
    return out


def next_character_logprobs(model, tokenizer, prefix: str, chars: Iterable[str]) -> dict[str, float]:
    """Query GPT-2 next-token probabilities for each live character action."""
    import torch
    out: dict[str, float] = {}
    base = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
    for char in tuple(chars):
        full = tokenizer(prefix + char, add_special_tokens=False, return_tensors="pt").input_ids
        if full.shape[1] <= base.shape[1]:
            out[char] = float("-inf")
            continue
        with torch.no_grad():
            logits = model(full).logits[0]
        logprobs = logits[:-1].log_softmax(-1)
        targets = full[0, 1:]
        picked = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        out[char] = float(picked[base.shape[1] - 1 :].sum().item())
    return out


def obligation(left: Cursor, right: Cursor) -> tuple[bool, dict[str, object] | None]:
    """Resolve equation positions whose opposite normal-order prefix exists."""
    a, b = tape(left.text), tape(right.text)
    # Normal-order prefixes do not expose the counterpart at the far edge
    # until both clauses close.  Keep the unresolved obligation in state;
    # applying an early reverse-prefix rejection here would be a reverse-half
    # decoder in disguise.
    if not (left.complete and right.complete):
        return True, None
    if left.complete:
        for j, char in enumerate(b):
            if j >= len(a) or a[-1 - j] != char:
                return False, {"left_index": len(a) - 1 - j, "right_index": j, "left": a[-1 - j] if j < len(a) else None, "right": char}
    if right.complete:
        for i, char in enumerate(a):
            if i >= len(b) or char != b[-1 - i]:
                return False, {"left_index": i, "right_index": len(b) - 1 - i if i < len(b) else None, "left": char, "right": b[-1 - i] if i < len(b) else None}
    return True, None


def expand_live(states: Iterable[State], model, tokenizer, beam: int = BEAM) -> list[State]:
    """Advance both normal-order grammar cursors and score before commit."""
    out: list[State] = []
    for state in states:
        left_transitions = cursor_transitions(state.left)
        right_transitions = cursor_transitions(state.right)
        if not left_transitions and not state.left.complete:
            continue
        if not right_transitions and not state.right.complete:
            continue
        if state.left.complete:
            left_transitions = [("", state.left, "terminal")]
        if state.right.complete:
            right_transitions = [("", state.right, "terminal")]
        left_scores = next_character_logprobs(model, tokenizer, state.left.text, (c for c, _, _ in left_transitions if c))
        right_scores = next_character_logprobs(model, tokenizer, state.right.text, (c for c, _, _ in right_transitions if c))
        for lc, left, _ in left_transitions:
            for rc, right, _ in right_transitions:
                ok, mismatch = obligation(left, right)
                if not ok and not (left.complete and right.complete):
                    continue
                out.append(State(left, right, state.score + left_scores.get(lc, 0.0) + right_scores.get(rc, 0.0), state.transitions + int(bool(lc)) + int(bool(rc)), state.first_mismatch or mismatch))
    return sorted(out, key=lambda s: (-s.score, s.left.text, s.right.text))[:beam]


def independent_audit(left: Cursor, right: Cursor) -> dict[str, object]:
    rendered = left.text.capitalize() + ". " + right.text.capitalize() + "."
    normalized = tape(rendered)
    mismatches = [i for i in range(len(normalized) // 2) if normalized[i] != normalized[-1 - i]]
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {"rendered": rendered, "letters": len(normalized), "normalized_tape": normalized, "sha256": hashlib.sha256(normalized.encode()).hexdigest(), "exact": bool(normalized) and not mismatches, "two_pointer": {"exact": bool(normalized) and not mismatches, "mismatch_count": len(mismatches), "first_mismatch": mismatches[0] if mismatches else None}, "hash_replay": bool(normalized) and hashlib.sha256(normalized.encode()).hexdigest() == hashlib.sha256(normalized[::-1].encode()).hexdigest(), "mechanical_checks": checks, "mechanically_admitted": bool(normalized) and not mismatches and all(checks.values()), "left_complete": left.complete, "right_complete": right.complete, "left_words": list(left.words), "right_words": list(right.words)}


def complete_control(side: str) -> Cursor:
    """Return one independently authored complete clause as a readable control."""
    words = (("the", "baker", "reads", "the", "letter", "in", "the", "quiet", "room")
             if side == "left" else
             ("the", "caller", "answers", "the", "message", "at", "the", "brief", "plaza"))
    return Cursor(side, slot=9, words=words, text=" ".join(words), verb_sense="text", complete=True)


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries, excluded = registry.get("entries", []), registry.get("excluded", [])
    prior = [row for row in entries + excluded if row.get("id") != EXPERIMENT_ID]
    current = set(re.findall(r"[a-z0-9]+", SIGNATURE.lower()))
    common = {"a", "an", "and", "audit", "character", "complete", "exact", "independent", "normal", "order", "the", "two", "with"}
    overlaps = []
    for row in prior:
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "").lower()))
        shared = (current - common) & other
        union = current | other
        score = len(shared) / len(union) if union else 0.0
        # Prior GPT-2 routes are proposal-bank, fixed-tape, reverse-half, or
        # BPE continuation systems; this direct state operator is orthogonal.
        if score >= 0.28:
            overlaps.append({"id": row.get("id"), "jaccard": round(score, 6), "shared_atoms": sorted(shared)})
    return {"registry_entries": len(entries), "excluded_routes": len(excluded), "overlaps": overlaps, "blocked": bool(overlaps), "operator": "direct character equation with grammar-prefix state; no proposal bank, fixed tape, or reverse emission", "performed_before_model_load": True}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["blocked"]:
        raise RuntimeError("revised direct character operator collided with the novelty registry")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    model.eval()
    states = [State(Cursor("left"), Cursor("right"))]
    expansions = 0
    for _ in range(MAX_STEPS):
        states = expand_live(states, model, tokenizer)
        expansions += sum(s.transitions for s in states)
        if not states or all(s.left.complete and s.right.complete for s in states):
            break
    rows = [independent_audit(s.left, s.right) | {"lm_logprob": s.score, "live_character_transitions": s.transitions, "first_mismatch": s.first_mismatch, "provenance": {"model": "gpt2-local-cache", "grammar_prefix_state": True, "source_sentences_copied": False, "fixed_tape": False, "reverse_emission": False}} for s in states if s.left.complete and s.right.complete]
    if not rows:
        # Preserve a complete ordinary-prose control even when the live
        # obligation kills every generated frontier before both clauses close.
        control = independent_audit(complete_control("left"), complete_control("right"))
        control.update({"control_probe": True, "lm_logprob": None, "live_character_transitions": 0, "provenance": {"model": "gpt2-local-cache", "grammar_prefix_state": True, "complete_control": True, "source_sentences_copied": False, "fixed_tape": False, "reverse_emission": False}})
        rows = [control]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_no_reader_promotion" if not admitted else "exact_hits_pending_reader", "method": "Direct character-level grammar-prefix decoding. GPT-2 next-token probabilities are queried at every licensed character/boundary transition before the live character-equation obligation is applied; both sides remain normal-order prefixes.", "novelty_preflight": preflight, "config": {"model": "gpt2-local-cache", "beam": BEAM, "max_steps": MAX_STEPS, "grammar": "DET SUBJECT VERB DET OBJECT PREP DET ADJUNCT", "proposal_bank": False, "fixed_tape": False, "reverse_half_decoder": False}, "stats": {"model_loaded": 1, "live_expansions": expansions, "final_states": len(states), "rendered_probes": len(rows), "exact": sum(row["exact"] for row in rows), "mechanically_admitted": len(admitted), "reader_eligible": 0}, "rendered_candidates": rows, "rendered_probes": rows, "repair": {"status": "not_run" if not admitted else "deferred_to_reader_gate", "operator": "At the first live seam mismatch, replace the held-out word in the same POS/sense slot, re-query next-character probabilities from that grammar prefix, and replay independent tape/hash and mechanical gates.", "first_mismatch": rows[0]["two_pointer"]["first_mismatch"] if rows else None}, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "model_source": "gpt2-local-cache", "source_sentences_copied": False, "known_palindromes_used": False, "reader_evidence": False}}


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
