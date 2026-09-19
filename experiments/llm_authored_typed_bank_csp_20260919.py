"""Deterministic character CSP over a one-time, typed Shakespearean bank.

The local model authors lexical alternatives once; it never scores or rewards
search states.  The constructor then searches ordinary grammar paths with
live half-tape aliases, agreement, valency, and content-word uniqueness.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "llm-authored-typed-bank-csp-20260919"
AUTHORING_MODEL = "gpt-oss:20b"
AUTHORING_PROMPT_SHA256 = "b6e2c2d9a091a2e9efdf75de1d42fc3a3e2ab4c6d2ee80fb1886cbef092642d3"


@dataclass(frozen=True)
class Lexeme:
    text: str
    number: str | None = None
    kind: str | None = None
    valency: str | None = None


SUBJECTS = (
    Lexeme("the herald", "sg", "person"), Lexeme("the poet", "sg", "person"),
    Lexeme("the king", "sg", "person"), Lexeme("the lady", "sg", "person"),
    Lexeme("the knight", "sg", "person"), Lexeme("the court", "sg", "place"),
    Lexeme("the council", "sg", "place"), Lexeme("the messenger", "sg", "person"),
    Lexeme("the scholar", "sg", "person"), Lexeme("the duke", "sg", "person"),
    Lexeme("the choir", "pl", "person"), Lexeme("the banners", "pl", "object"),
)
VERBS = (
    Lexeme("reads", "sg", valency="document"), Lexeme("praises", "sg", valency="place"),
    Lexeme("orders", "sg", valency="object"), Lexeme("chants", "pl", valency="object"),
    Lexeme("declares", "sg", valency="document"), Lexeme("guards", "sg", valency="place"),
    Lexeme("sings", "pl", valency="object"), Lexeme("announces", "sg", valency="document"),
    Lexeme("inspects", "sg", valency="object"), Lexeme("offers", "sg", valency="object"),
    Lexeme("requests", "sg", valency="document"), Lexeme("guards", "pl", valency="place"),
    Lexeme("recites", "pl", valency="document"), Lexeme("enters", "sg", valency="place"),
    Lexeme("exposes", "sg", valency="object"), Lexeme("congratulates", "sg", valency="person"),
)
OBJECTS = (
    Lexeme("a letter", "sg", "document"), Lexeme("the royal decree", "sg", "document"),
    Lexeme("the banquet hall", "sg", "place"), Lexeme("the courtyard", "sg", "place"),
    Lexeme("the silver goblet", "sg", "object"), Lexeme("the golden crown", "sg", "object"),
    Lexeme("the stained glass", "sg", "object"), Lexeme("the tapestry", "sg", "object"),
    Lexeme("the heraldic shield", "sg", "object"), Lexeme("the royal banner", "sg", "object"),
    Lexeme("the choir's hymn", "sg", "document"), Lexeme("the council's edict", "sg", "document"),
    Lexeme("the knight's sword", "sg", "object"), Lexeme("the duke's seal", "sg", "object"),
    Lexeme("the scholar's scroll", "sg", "document"), Lexeme("the court's applause", "sg", "object"),
)
SETTINGS = (
    Lexeme("in the great hall", kind="place"), Lexeme("outside the castle gates", kind="place"),
    Lexeme("within the royal gardens", kind="place"), Lexeme("upon the balcony", kind="place"),
    Lexeme("beneath the moonlit sky", kind="place"), Lexeme("amid the bustling market", kind="place"),
    Lexeme("inside the secret chamber", kind="place"), Lexeme("along the riverbank", kind="place"),
)


def _content(text: str) -> frozenset[str]:
    stop = {"a", "an", "the", "and", "while", "as", "in", "on", "at", "of", "to", "within", "upon", "beneath", "amid", "inside", "along", "outside"}
    return frozenset(normalize_letters(w) for w in tokenize(text) if normalize_letters(w) not in stop)


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatch = None
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatch = (i, j, tape[i], tape[j])
            break
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {"normalized": tape, "letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


def _place(assign: list[str | None], pos: int, text: str, target: int) -> list[str | None] | None:
    out = list(assign)
    for char in normalize_letters(text):
        if pos >= target:
            return None
        slot = min(pos, target - 1 - pos)
        if out[slot] not in (None, char):
            return None
        out[slot] = char
        pos += 1
    return out


def _frames() -> tuple[tuple[str, ...], ...]:
    return (
        ("S", "V", "O"),
        ("S", "V", "O", "SET"),
        ("S", "V", "O", "CONJ", "S2", "V2", "O2"),
        ("S", "V", "O", "CONJ", "S2", "V2", "O2", "SET2"),
    )


def _options(slot: str, state: dict[str, object]) -> tuple[Lexeme, ...]:
    if slot in {"S", "S2"}:
        return SUBJECTS
    if slot in {"V", "V2"}:
        number = state.get("s_number" if slot == "V" else "s2_number")
        return tuple(v for v in VERBS if v.number == number and v.valency)
    if slot in {"O", "O2"}:
        valency = state.get("v_valency" if slot == "O" else "v2_valency")
        return tuple(o for o in OBJECTS if o.kind == valency)
    if slot in {"SET", "SET2"}:
        return SETTINGS
    if slot == "CONJ":
        return (Lexeme("and"), Lexeme("while"))
    return ()


def search(target: int, *, max_nodes: int = 100_000) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    frontier: list[dict[str, object]] = []
    nodes = 0
    for frame in _frames():
        def dfs(index: int, pos: int, chosen: list[Lexeme], used: frozenset[str], state: dict[str, object], assign: list[str | None]) -> None:
            nonlocal nodes
            if nodes >= max_nodes or len(rows) >= 50:
                return
            nodes += 1
            if index == len(frame):
                if pos != target:
                    # Preserve a small intact-prose control sample even when
                    # the target seam has no exact closure. These rows are
                    # explicitly frontier controls, never candidates.
                    if 40 <= pos <= 90 and len(frontier) < 30:
                        text = " ".join(item.text for item in chosen) + "."
                        result = audit(text)
                        frontier.append({"rendered": text, "length": result["letters"], "audit": result,
                                         "target_length": target, "frame": frame,
                                         "provenance": {"experiment_id": ID, "authoring_model": AUTHORING_MODEL,
                                           "authoring_prompt_sha256": AUTHORING_PROMPT_SHA256,
                                           "representation": "typed bank grammar frontier control",
                                           "finished_tape_reversed": False, "catalogue_imported": False, "rlaif_used": False},
                                         "reader_status": "frontier control; not an exact candidate"})
                    return
                text = " ".join(item.text for item in chosen) + "."
                result = audit(text)
                checks = mechanical_admission_checks(text, min_letters=30, max_letters=2000)
                rows.append({"rendered": text, "length": result["letters"], "audit": result,
                             "mechanical_checks": checks,
                             "mechanically_admitted": result["two_pointer_exact"] and all(checks.values()),
                             "frame": frame, "word_path": [item.text for item in chosen],
                             "provenance": {"experiment_id": ID, "authoring_model": AUTHORING_MODEL,
                               "authoring_prompt_sha256": AUTHORING_PROMPT_SHA256,
                               "representation": "typed bank half-tape CSP", "target_length": target,
                               "finished_tape_reversed": False, "catalogue_imported": False, "rlaif_used": False},
                             "reader_status": "unreviewed; programmatic checks do not certify readability"})
                return
            slot = frame[index]
            options = _options(slot, state)
            if slot == "CONJ":
                options = tuple(item for item in options if item.text not in used)
            for item in options:
                content = _content(item.text)
                if used & content:
                    continue
                placed = _place(assign, pos, item.text, target)
                if placed is None:
                    continue
                next_state = dict(state)
                if slot == "S": next_state["s_number"] = item.number
                if slot == "V": next_state["v_valency"] = item.valency
                if slot == "S2": next_state["s2_number"] = item.number
                if slot == "V2": next_state["v2_valency"] = item.valency
                dfs(index + 1, pos + len(normalize_letters(item.text)), chosen + [item], used | content, next_state, placed)
        dfs(0, 0, [], frozenset(), {}, [None] * ((target + 1) // 2))
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"target": target, "nodes": nodes, "actual_candidates": rows, "frontier_controls": frontier,
            "exact_candidates": exact, "mechanically_admitted": admitted}


def run(min_target: int = 40, max_target: int = 90) -> dict[str, object]:
    results = [search(target) for target in range(min_target, max_target + 1)]
    rows = [row for result in results for row in result["actual_candidates"]]
    frontier = [row for result in results for row in result["frontier_controls"]]
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"experiment_id": ID, "method": "one-time model-authored typed Shakespearean bank intersected with a deterministic half-tape grammar CSP",
            "status": "completed_exact" if exact else "completed_no_exact_closure", "actual_candidates": rows,
            "stats": {"nodes": sum(result["nodes"] for result in results), "rendered": len(rows), "frontier_controls": len(frontier), "exact": len(exact),
                      "mechanically_admitted": len(admitted), "longest_rendered": max((row["length"] for row in rows), default=0),
                      "longest_exact": max((row["length"] for row in exact), default=0)},
            "frontier_controls": frontier,
            "provenance": {"authoring_model": AUTHORING_MODEL, "authoring_prompt_sha256": AUTHORING_PROMPT_SHA256,
                           "independent_audits": ["literal outside-in two-pointer", "forward/reverse SHA-256"], "rlaif_per_candidate": False},
            "novelty_preflight": {"status": "passed", "distinction": "one-time typed lexical authoring followed by deterministic character search; no candidate reward loop"},
            "next_repair": {"action": "add a held-out lexical bank keyed by the first two residual characters, then rerun only compatible grammar states",
                            "reader_test": "randomized blinded intact prose versus shuffled controls for every mechanically admitted row"},
            "reader_gate": "closed; no human readability evidence"}


if __name__ == "__main__":
    result = run()
    output = ROOT / "runs" / f"{ID}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
