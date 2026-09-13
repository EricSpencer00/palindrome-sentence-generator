"""Bounded whole-prose repair pilot for original readable English palindromes.

This is a deliberate replacement for the bilateral-tape pilot.  Each search
state is a complete, independently readable *draft* and may have many
symmetry errors.  A proposer is asked to rewrite that full prose while
preserving its communicative intent; it is never asked to provide mirrored
halves and this program never supplies counterpart letters or a rendering.

Only a zero-mismatch authored surface is passed to the central mechanical
admission gate and a separately implemented exactness check.  The prose and
critique signals in this module are triage evidence, not a claim that a model
has established readability.  A blinded human study remains downstream of an
eligible 100--160-letter candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


DATE = "20260913"
LINEAGES = 4
REPAIR_ROUNDS = 8
ALTERNATIVES_PER_REPAIR = 4
MIN_LETTERS = 100
MAX_LETTERS = 160
DEFAULT_SEED = 20260927
HOST = "http://localhost:11434"
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


class LocalProseClient(Protocol):
    """The small, mockable boundary around an explicitly local proposer."""

    def metadata(self, model: str) -> dict[str, Any]: ...

    def complete(self, *, model: str, prompt: str, seed: int) -> str: ...


class OllamaLocalClient:
    """Local client used only after the CLI caller explicitly supplies --live."""

    def __init__(self, host: str = HOST):
        self.host = host.rstrip("/")

    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import urllib.request

        request = urllib.request.Request(
            self.host + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def metadata(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"name": model})

    def complete(self, *, model: str, prompt: str, seed: int) -> str:
        response = self._request(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": "low",
                "options": {"temperature": 0.7, "seed": seed},
            },
        )
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("local_response_has_no_message_content")
        return content


class ReplayClient:
    """Filesystem-only client for deterministic tests and recorded replays."""

    def __init__(self, replies: list[str], metadata: dict[str, Any] | None = None):
        self.replies = list(replies)
        self._metadata = metadata or {"backend": "replay"}

    def metadata(self, model: str) -> dict[str, Any]:
        return {**self._metadata, "requested_model": model}

    def complete(self, *, model: str, prompt: str, seed: int) -> str:
        if not self.replies:
            raise ValueError("replay_has_too_few_replies")
        return self.replies.pop(0)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def independent_ascii_letters(text: str) -> str:
    """Independent ASCII-only normalizer, deliberately separate from admission."""
    if not isinstance(text, str):
        raise ValueError("rendering_is_not_a_string")
    if any(char.isalpha() and not char.isascii() for char in text):
        raise ValueError("non_ascii_alphabetic_character")
    return "".join(char.lower() for char in text if "A" <= char <= "Z" or "a" <= char <= "z")


def _normalized_word_map(text: str) -> list[str | None]:
    """Map each independently normalized character to its containing word."""
    result: list[str | None] = []
    cursor = 0
    for match in WORD_RE.finditer(text.casefold()):
        while cursor < match.start():
            for char in text[cursor:match.start()]:
                if char.isascii() and char.isalpha():
                    result.append(None)
            cursor = match.start()
        word = match.group()
        for char in word:
            if char.isascii() and char.isalpha():
                result.append(word)
        cursor = match.end()
    for char in text[cursor:]:
        if char.isascii() and char.isalpha():
            result.append(None)
    return result


def symmetry_diagnostics(text: str) -> dict[str, Any]:
    """Expose character mismatches and their words without deciding prose quality."""
    tape = independent_ascii_letters(text)
    word_map = _normalized_word_map(text)
    if len(word_map) != len(tape):
        raise AssertionError("word_map_and_normalized_tape_diverged")
    pairs = []
    for left in range(len(tape) // 2):
        right = len(tape) - 1 - left
        if tape[left] != tape[right]:
            pairs.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left_letter": tape[left],
                    "right_letter": tape[right],
                    "left_word": word_map[left],
                    "right_word": word_map[right],
                }
            )
    return {
        "normalized_tape": tape,
        "letters": len(tape),
        "mismatch_count": len(pairs),
        "mismatch_rate": len(pairs) / max(1, len(tape) // 2),
        "mismatches": pairs,
    }


def independent_exactness(text: str) -> dict[str, Any]:
    """Second exactness implementation stored with every closure audit."""
    tape = independent_ascii_letters(text)
    return {
        "normalizer": "explicit_ascii_scan_v1",
        "tape": tape,
        "letter_count": len(tape),
        "direct_symmetric_position_comparison": all(tape[index] == tape[-1 - index] for index in range(len(tape))),
    }


def non_exact_mechanical_screen(text: str) -> dict[str, bool]:
    """Apply every central non-exact hard screen while retaining errorful prose."""
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {key: value for key, value in checks.items() if key != "exact_letter_palindrome"}


@dataclass(frozen=True)
class Draft:
    lineage: int
    round_index: int
    text: str
    intent: str
    parent_sha256: str | None
    provenance: dict[str, Any]

    def record(self) -> dict[str, Any]:
        diagnostics = symmetry_diagnostics(self.text)
        screen = non_exact_mechanical_screen(self.text)
        return {
            **asdict(self),
            "text_sha256": _digest(self.text),
            "symmetry_diagnostics": diagnostics,
            "non_exact_mechanical_screen": screen,
            "passes_non_exact_mechanical_screen": all(screen.values()),
        }


def initialization_prompt() -> str:
    """Ask for complete original scenes without supplying palindrome material."""
    payload = {
        "task": "Write four original, self-contained English passages.",
        "required_scene_kinds": [
            "a small action with a consequence",
            "an observation with an explanation",
            "an instruction with a reason",
            "a correction of a misunderstanding",
        ],
        "hard_rules": [
            "Return one JSON object and nothing else.",
            "Return exactly four drafts in the listed scene-kind order.",
            "Each text must be a coherent complete thought that makes sense by itself.",
            "Each text must contain 100 to 160 ASCII letters after spaces and punctuation are removed.",
            "Use ordinary English words; no lists, fragments, repeated content words, quotations, named catalogue material, or wordplay.",
            "Do not discuss constraints or the writing process in a text.",
        ],
        "response_schema": {
            "drafts": [
                {"intent": "one sentence describing the scene's proposition", "text": "complete English passage"}
            ]
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def repair_prompt(draft: Draft) -> str:
    """Present whole prose, intent, and indexed mismatches—not synthetic halves."""
    diagnostics = symmetry_diagnostics(draft.text)
    payload = {
        "task": "Rewrite a complete English passage into four different complete English passages expressing the same event or proposition.",
        "communicative_intent": draft.intent,
        "current_complete_prose": draft.text,
        "character_diagnostics": diagnostics,
        "hard_rules": [
            "Return one JSON object and nothing else.",
            "Return exactly four alternatives, each with a complete text and a brief notes field.",
            "Every alternative must make sense without the supplied explanation.",
            "You may change every word, punctuation mark, word boundary, and total length; keep 100 to 160 ASCII letters after normalization.",
            "Preserve an ordinary, sensible version of the intent while repairing several interacting letter conflicts together.",
            "Aim to make letters at corresponding positions from opposite ends agree, but do not split the prose into halves, emit reversed strings, use palindrome examples, repeated content words, lists, fragments, or wordplay.",
            "Do not claim the output is readable or exact; the host verifies mechanical properties.",
        ],
        "response_schema": {"alternatives": [{"text": "complete English passage", "notes": "brief revision intent"}]},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def critique_prompt(drafts: list[Draft]) -> str:
    """Separate surface-only triage: intentionally no symmetry metrics or scores."""
    payload = {
        "task": "Flag prose defects in each complete passage. This is triage only, not a readability judgment or a rewrite request.",
        "items": [{"lineage": draft.lineage, "round": draft.round_index, "intent": draft.intent, "text": draft.text} for draft in drafts],
        "defects_to_check": ["fragment", "missing argument", "incoherent reference", "implausible predicate", "meaning drift"],
        "hard_rules": ["Return one JSON object and nothing else.", "Return one result per input in the same order.", "Do not mention letter patterns, symmetry, scores, or palindromes."],
        "response_schema": {"items": [{"lineage": 0, "round": 0, "defects": ["zero or more rubric labels"]}]},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _strict_json(raw: str, outer: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"reply_is_not_strict_json:{error.msg}") from error
    if not isinstance(value, dict) or set(value) != {outer} or not isinstance(value[outer], list):
        raise ValueError(f"reply_must_contain_only_{outer}_array")
    return value[outer]


def parse_initialization(raw: str) -> tuple[list[dict[str, str]] | None, str | None]:
    try:
        rows = _strict_json(raw, "drafts")
        if len(rows) != LINEAGES:
            raise ValueError("initialization_requires_exactly_four_drafts")
        if any(not isinstance(row, dict) or set(row) != {"intent", "text"} or not all(isinstance(row[key], str) and row[key].strip() for key in row) for row in rows):
            raise ValueError("initialization_draft_schema_invalid")
        return rows, None
    except ValueError as error:
        return None, str(error)


def parse_repairs(raw: str) -> tuple[list[dict[str, str]] | None, str | None]:
    try:
        rows = _strict_json(raw, "alternatives")
        if len(rows) != ALTERNATIVES_PER_REPAIR:
            raise ValueError("repair_requires_exactly_four_alternatives")
        if any(not isinstance(row, dict) or set(row) != {"notes", "text"} or not all(isinstance(row[key], str) and row[key].strip() for key in row) for row in rows):
            raise ValueError("repair_alternative_schema_invalid")
        if len({row["text"] for row in rows}) != len(rows):
            raise ValueError("repair_alternatives_must_be_distinct")
        return rows, None
    except ValueError as error:
        return None, str(error)


def parse_critiques(raw: str, drafts: list[Draft]) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        rows = _strict_json(raw, "items")
        expected = [(draft.lineage, draft.round_index) for draft in drafts]
        seen = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"defects", "lineage", "round"}:
                raise ValueError("critique_item_schema_invalid")
            if not isinstance(row["lineage"], int) or not isinstance(row["round"], int) or not isinstance(row["defects"], list) or not all(isinstance(defect, str) for defect in row["defects"]):
                raise ValueError("critique_item_types_invalid")
            seen.append((row["lineage"], row["round"]))
        if seen != expected:
            raise ValueError("critique_items_do_not_match_selected_drafts")
        return rows, None
    except ValueError as error:
        return None, str(error)


def _rank(draft: Draft) -> tuple[Any, ...]:
    record = draft.record()
    diagnostics = record["symmetry_diagnostics"]
    return (
        not record["passes_non_exact_mechanical_screen"],
        diagnostics["mismatch_rate"],
        diagnostics["mismatch_count"],
        record["text_sha256"],
    )


def select_revision(candidates: list[Draft], *, round_index: int, parent: Draft) -> tuple[Draft, dict[str, Any]]:
    """Frozen selection: target length is invariant; two rounds explore alignment."""
    in_band = [candidate for candidate in candidates if MIN_LETTERS <= independent_exactness(candidate.text)["letter_count"] <= MAX_LETTERS]
    # A paraphrase that drops below the paper floor is not an exploratory state.
    # Keep the last valid full passage instead of rewarding a shorter mismatch
    # rate; otherwise the previous pilot literally selected its way into
    # fragments.
    if not in_band:
        return parent, {
            "rule": "retain_parent_when_no_alternative_is_in_target_length_band",
            "round_index": round_index,
            "chosen_index": None,
            "exploratory": False,
            "ranked_text_sha256": [_digest(candidate.text) for candidate in candidates],
        }
    ordered = sorted(in_band, key=_rank)
    exploratory = round_index in {2, 5} and len(ordered) > 1
    chosen_index = 1 if exploratory else 0
    return ordered[chosen_index], {
        "rule": "rank_nonexact_screen_then_mismatch_rate_then_mismatch_count_then_hash; ranks_1_on_rounds_2_and_5_exploratorily",
        "round_index": round_index,
        "chosen_index": chosen_index,
        "exploratory": exploratory,
        "ranked_text_sha256": [_digest(candidate.text) for candidate in ordered],
    }


def candidate_audit(draft: Draft) -> dict[str, Any]:
    """Freeze exact/admission evidence for an authored full-prose surface."""
    central = mechanical_admission_checks(draft.text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    independent = independent_exactness(draft.text)
    same_tape = independent["tape"] == normalize_letters(draft.text)
    eligible = all(central.values()) and independent["direct_symmetric_position_comparison"] and same_tape
    return {
        "rendered": draft.text,
        "render_sha256": _digest(draft.text),
        "letters": independent["letter_count"],
        "central_mechanical_checks": central,
        "independent_exactness": independent,
        "independent_and_central_tapes_agree": same_tape,
        "mechanically_eligible": eligible,
        "provenance": draft.provenance,
        "human_reader_study": "not_run",
    }


def _call(client: LocalProseClient, *, model: str, prompt: str, seed: int) -> dict[str, Any]:
    record = {"model": model, "seed": seed, "prompt": prompt, "prompt_sha256": _digest(prompt)}
    try:
        raw = client.complete(model=model, prompt=prompt, seed=seed)
        if not isinstance(raw, str):
            raise ValueError("client_reply_not_string")
        record.update({"raw_reply": raw, "reply_sha256": _digest(raw)})
    except (OSError, ValueError, json.JSONDecodeError) as error:
        record["driver_rejection"] = str(error)
    return record


def run_whole_prose_repair_pilot(client: LocalProseClient, *, model: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Run exactly one initial call, 32 repair calls, and 8 surface-only critiques."""
    metadata = client.metadata(model)
    ledger: dict[str, Any] = {
        "status": "whole_prose_repair_pilot_complete",
        "config": {
            "date": DATE,
            "lineages": LINEAGES,
            "repair_rounds_per_lineage": REPAIR_ROUNDS,
            "alternatives_per_repair": ALTERNATIVES_PER_REPAIR,
            "candidate_letter_range": [MIN_LETTERS, MAX_LETTERS],
            "construction_representation": "complete_prose_with_temporary_symmetry_errors",
            "counterpart_synthesis": "forbidden",
            "word_order_mirror_construction": "forbidden",
            "machine_readability_certification": False,
            "fixed_call_budget": {"initialization": 1, "repair": LINEAGES * REPAIR_ROUNDS, "critique": REPAIR_ROUNDS},
        },
        "model_metadata": metadata,
        "initialization": None,
        "repair_calls": [],
        "critique_calls": [],
        "candidate_audits": [],
        "eligible_100_plus_candidates": [],
    }
    initial_seed = seed
    initialization = _call(client, model=model, prompt=initialization_prompt(), seed=initial_seed)
    rows, rejection = parse_initialization(initialization.get("raw_reply", "")) if "raw_reply" in initialization else (None, initialization.get("driver_rejection"))
    initialization["schema_rejection"] = rejection
    initialization["parsed_drafts"] = rows
    ledger["initialization"] = initialization
    if rows is None:
        ledger["human_reader_study"] = {"triggered": False, "reason": "initialization_failed; no candidate exists"}
        return ledger
    current = [
        Draft(lineage=index, round_index=-1, text=row["text"], intent=row["intent"], parent_sha256=None,
              provenance={"kind": "model", "stage": "initialization", "seed": str(initial_seed), "reply_sha256": initialization["reply_sha256"], "lineage": index})
        for index, row in enumerate(rows)
    ]
    for round_index in range(REPAIR_ROUNDS):
        selected: list[Draft] = []
        for lineage in range(LINEAGES):
            parent = current[lineage]
            call_seed = seed + 1 + round_index * LINEAGES + lineage
            call = _call(client, model=model, prompt=repair_prompt(parent), seed=call_seed)
            call.update({"round_index": round_index, "lineage": lineage, "parent": parent.record()})
            rows, rejection = parse_repairs(call.get("raw_reply", "")) if "raw_reply" in call else (None, call.get("driver_rejection"))
            call["schema_rejection"] = rejection
            call["parsed_alternatives"] = rows
            alternatives: list[Draft] = []
            if rows is not None:
                for alternative_index, row in enumerate(rows):
                    alternatives.append(
                        Draft(
                            lineage=lineage,
                            round_index=round_index,
                            text=row["text"],
                            intent=parent.intent,
                            parent_sha256=_digest(parent.text),
                            provenance={"kind": "model", "stage": "repair", "seed": str(call_seed), "round": round_index,
                                        "lineage": lineage, "alternative_index": alternative_index,
                                        "reply_sha256": call["reply_sha256"], "notes": row["notes"], "parent_sha256": _digest(parent.text)},
                        )
                    )
                chosen, selection = select_revision(alternatives, round_index=round_index, parent=parent)
                call["alternatives"] = [alternative.record() for alternative in alternatives]
                call["selection"] = selection
                call["selected"] = chosen.record()
                selected.append(chosen)
                audit = candidate_audit(chosen)
                ledger["candidate_audits"].append(audit)
                if audit["mechanically_eligible"]:
                    ledger["eligible_100_plus_candidates"].append(audit)
            else:
                call["alternatives"] = []
                call["selected"] = parent.record()
                selected.append(parent)
            ledger["repair_calls"].append(call)
        current = selected
        critique_seed = seed + 1 + LINEAGES * REPAIR_ROUNDS + round_index
        critique = _call(client, model=model, prompt=critique_prompt(current), seed=critique_seed)
        rows, rejection = parse_critiques(critique.get("raw_reply", ""), current) if "raw_reply" in critique else (None, critique.get("driver_rejection"))
        critique.update({"round_index": round_index, "schema_rejection": rejection, "parsed_items": rows})
        ledger["critique_calls"].append(critique)
    ledger["human_reader_study"] = {
        "triggered": False,
        "reason": "no mechanically eligible 100+ exact candidate" if not ledger["eligible_100_plus_candidates"] else "candidate exists; human study has not been run",
    }
    ledger["source_provenance"] = {"driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "all_raw_prompts_and_replies_retained": True}
    return ledger


def _replay_client(path: Path) -> ReplayClient:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) - {"metadata", "replies"}:
        raise ValueError("replay_file_has_unknown_fields")
    replies = value.get("replies")
    expected = 1 + LINEAGES * REPAIR_ROUNDS + REPAIR_ROUNDS
    if not isinstance(replies, list) or len(replies) != expected or not all(isinstance(row, str) for row in replies):
        raise ValueError(f"replay_requires_exactly_{expected}_string_replies")
    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("replay_metadata_must_be_an_object")
    return ReplayClient(replies, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path, help="deterministic 41-reply JSON ledger")
    source.add_argument("--live", action="store_true", help="explicitly permit 41 local requests")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client: LocalProseClient = _replay_client(args.replay) if args.replay else OllamaLocalClient(args.host)
    report = run_whole_prose_repair_pilot(client, model=args.model, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "repair_calls": len(report["repair_calls"]), "critique_calls": len(report["critique_calls"]), "eligible_100_plus_candidates": len(report["eligible_100_plus_candidates"]), "human_reader_study_triggered": report["human_reader_study"]["triggered"]}, sort_keys=True))


if __name__ == "__main__":
    main()
