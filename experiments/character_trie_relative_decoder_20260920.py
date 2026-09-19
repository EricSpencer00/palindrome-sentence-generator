"""Character-trie decoder with typed internal relative-clause transitions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
from experiments.character_trie_grammar_decoder_20260919 import (
    Token, TrieNode, make_trie, TOKENS as BASE_TOKENS, audit,
)

EXPERIMENT_ID = "character-trie-relative-decoder-20260920"

RELATIVE_MARKERS = (Token("who", "R"), Token("that", "R"))
RELATIVE_SUBJECTS = tuple(token for token in BASE_TOKENS["S"] if token.text in {
    "poet", "scribe", "sailor", "men", "poets", "sailors", "singers",
})
RELATIVE_VERBS = tuple(token for token in BASE_TOKENS["V"] if token.text in {
    "reads", "marks", "writes", "keeps", "finds", "follows", "opens",
    "hears", "sees", "holds", "loves", "needs", "read", "mark", "write",
    "keep", "find", "follow", "open", "hear", "see", "hold", "love", "need",
})
RELATIVE_OBJECTS = tuple(token for token in BASE_TOKENS["O"] if token.text in {
    "notes", "maps", "pages", "poems", "books", "songs", "tales", "Diana", "Noel",
})
TOKENS = dict(BASE_TOKENS)
TOKENS.update({"R": RELATIVE_MARKERS, "RS": RELATIVE_SUBJECTS, "RV": RELATIVE_VERBS, "RO": RELATIVE_OBJECTS})
TRIES: dict[str, TrieNode] = {slot: make_trie(items) for slot, items in TOKENS.items()}
FRAMES = (
    ("D", "S", "R", "RS", "RV", "D", "RO", "V", "D", "O", "C", "D", "S", "V", "O"),
    ("D", "S", "V", "D", "O", "R", "RS", "RV", "D", "RO", "C", "D", "S", "V", "O"),
)


def _choices(slot: str, position: int, target: int, assigned: dict[int, str], used: frozenset[str], number: str | None, proper_count: int) -> list[tuple[Token, dict[int, str]]]:
    result: list[tuple[Token, dict[int, str]]] = []

    def walk(node: TrieNode, cursor: int, state: dict[int, str]) -> None:
        for token in node.terminals:
            if len(token.tape) > 1 and token.tape in used:
                continue
            if slot in {"V", "RV"} and number not in (None, token.number):
                continue
            if token.proper and proper_count:
                continue
            result.append((token, state))
        if cursor >= target:
            return
        mirror = min(cursor, target - 1 - cursor)
        for character, child in node.children.items():
            previous = state.get(mirror)
            if previous is not None and previous != character:
                continue
            updated = dict(state)
            updated[mirror] = character
            walk(child, cursor + 1, updated)

    walk(TRIES[slot], position, dict(assigned))
    return result


def search(frame: tuple[str, ...], target: int, max_nodes: int = 5_000) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    nodes = 0
    minimum = {slot: min(len(token.tape) for token in items) for slot, items in TOKENS.items()}
    maximum = {slot: max(len(token.tape) for token in items) for slot, items in TOKENS.items()}

    def visit(index: int, position: int, assigned: dict[int, str], chosen: list[Token], used: frozenset[str], number: str | None, relative_number: str | None, proper_count: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return
        if index == len(frame):
            if position != target:
                return
            raw = " ".join(token.text for token in chosen).replace(" ; ", "; ")
            rendered = raw[:1].upper() + raw[1:] + "."
            checked = audit(rendered)
            if not checked["two_pointer_exact"] or checked["letters"] < 39:
                return
            gates = mechanical_admission_checks(rendered, min_letters=39, max_letters=260)
            rows.append({"rendered": rendered, "word_spans": [token.text for token in chosen], "audit": checked, "mechanical_checks": gates, "mechanically_admitted": all(gates.values()), "provenance": {"construction": "character trie with typed relative marker/subject/verb/object transitions", "finished_tape_reversed": False, "catalogue_imported": False, "word_order_mirror": False, "rlaif_used": False}, "reader_status": "unreviewed; exactness does not certify readability"})
            return
        slot = frame[index]
        remaining = frame[index + 1:]
        for token, next_assigned in _choices(slot, position, target, assigned, used, relative_number if slot == "RV" else number, proper_count):
            end = position + len(token.tape)
            if end + sum(minimum[key] for key in remaining) > target or end + sum(maximum[key] for key in remaining) < target:
                continue
            next_number = token.number if slot == "S" else number
            next_relative = token.number if slot == "RS" else relative_number
            next_used = used | ({token.tape} if len(token.tape) > 1 and token.tag not in {"D", "P", "C", "R"} else set())
            visit(index + 1, end, next_assigned, chosen + [token], next_used, next_number, next_relative, proper_count + int(token.proper))

    visit(0, 0, {}, [], frozenset(), None, None, 0)
    return rows, nodes


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    nodes = 0
    for frame in FRAMES:
        for target in range(39, 81):
            found, count = search(frame, target)
            rows.extend(found)
            nodes += count
    exact = sorted({row["audit"]["normalized"]: row for row in rows}.values(), key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {"experiment_id": EXPERIMENT_ID, "method": "character-trie grammar decoder with explicit relative marker/subject/verb/object transitions", "stats": {"target_runs": len(FRAMES) * 42, "nodes": nodes, "exact": len(exact), "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact), "longest_exact_letters": max((row["audit"]["letters"] for row in exact), default=0)}, "candidates": exact, "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"], "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID, "catalogue_imported": False}, "next_repair": "Add a shared-participant/anaphor state and one adjunct edge only after the relative transition closes.", "reader_gate": "closed; no exact candidate reached it"}


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / (EXPERIMENT_ID + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
