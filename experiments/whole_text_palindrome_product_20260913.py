"""Exact palindrome construction over one complete finite text grammar.

Unlike comparing a question with a reversed answer, this leaves the midpoint
free: it can fall inside any word, clause, or discourse turn. This module
certifies only paths through the supplied grammar and character symmetry.
Semantic adequacy, originality, and reader acceptance are separate checks.

Each grammar is a product of finite lexical slots. The implementation expands
word choices into real character edges, without enumerating complete texts.
Different semantic plans must be compiled separately; sharing slots is not a
license to mix incompatible participants or semantic roles.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str
    # A boundary marker is carried by the final real character of a word.
    # It is not an epsilon transition and never adds letters.
    completed_word: str | None = None
    slot: int | None = None


@dataclass
class Grammar:
    start: int
    end: int
    edges: tuple[Edge, ...]
    slots: tuple[tuple[str, ...], ...]


def compile_slots(slots: tuple[tuple[str, ...], ...]) -> Grammar:
    if not slots or any(not choices for choices in slots):
        raise ValueError("Nonempty lexical slots are required")
    edges = []
    node_count = 1
    cursor = 0
    normalized = []
    for slot, raw_choices in enumerate(slots):
        choices = tuple(dict.fromkeys(raw_choices))
        if any(not word or any(c not in "abcdefghijklmnopqrstuvwxyz" for c in word) for word in choices):
            raise ValueError("Choices must be nonempty lowercase ASCII words")
        normalized.append(choices)
        end = node_count
        node_count += 1
        for word in choices:
            source = cursor
            for index, char in enumerate(word):
                final = index == len(word) - 1
                target = end if final else node_count
                if not final:
                    node_count += 1
                edges.append(Edge(source, target, char, word if final else None, slot if final else None))
                source = target
        cursor = end
    return Grammar(0, cursor, tuple(edges), tuple(normalized))


def replay_path(grammar: Grammar, path: tuple[Edge, ...]) -> dict:
    cursor = grammar.start
    words = []
    slots = []
    for edge in path:
        if edge not in grammar.edges or edge.source != cursor:
            return {"ok": False, "reason": "not_a_connected_grammar_path"}
        cursor = edge.target
        if edge.completed_word is not None:
            words.append(edge.completed_word)
            slots.append(edge.slot)
    tape = "".join(edge.char for edge in path)
    lexical_replay = (
        slots == list(range(len(grammar.slots)))
        and all(word in grammar.slots[i] for i, word in enumerate(words))
        and "".join(words) == tape
    )
    return {"ok": cursor == grammar.end and lexical_replay,
            "words": words, "tape": tape,
            "exact": all(tape[i] == tape[len(tape) - 1 - i] for i in range(len(tape) // 2)),
            "letters": len(tape)}


def construct(grammar: Grammar, max_states: int = 100_000) -> dict:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    # Reachability makes crossing the middle impossible. IDs are not a
    # topological ordering because each lexical alternative has private nodes.
    reachable_cache = {}
    def reachable(node):
        if node not in reachable_cache:
            result = {node}
            for edge in outgoing[node]:
                result.update(reachable(edge.target))
            reachable_cache[node] = result
        return reachable_cache[node]

    reachable(grammar.start)
    stack = [(grammar.start, grammar.end, (), ())]
    count = 0
    records = []
    seen = set()
    deepest = 0
    while stack and count < max_states:
        left, right, prefix, suffix = stack.pop()
        count += 1
        deepest = max(deepest, len(prefix))
        middle_options = [()] if left == right else []
        middle_options += [(edge,) for edge in outgoing[left] if edge.target == right]
        for middle in middle_options:
            path = prefix + middle + suffix
            replay = replay_path(grammar, path)
            if not replay["ok"] or not replay["exact"]:
                raise AssertionError("Product emitted an invalid path")
            key = tuple(replay["words"])
            if key not in seen:
                seen.add(key)
                records.append({**replay, "center_characters": len(middle),
                                "midpoint_letter_offset": len(prefix)})
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char == last.char and last.source in reachable(first.target):
                    stack.append((first.target, last.source, prefix + (first,), (last,) + suffix))
    return {"states": count, "pending_states": len(stack),
            "states_exhausted": not stack, "truncated": bool(stack),
            "deepest_matched_pairs": deepest, "records": records,
            "scope": "Only the supplied finite slot grammar; no readability or originality claim"}


def exhaustive_reference(slots):
    """Tiny-grammar test oracle, intentionally independent of the product."""
    result = set()
    for words in product(*slots):
        tape = "".join(words)
        if tape == tape[::-1]:
            result.add(words)
    return result
