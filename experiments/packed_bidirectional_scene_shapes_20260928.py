"""Packed intersection of two *different* ordinary-English sentence shapes.

The left and right sentence grammars are compiled independently.  A product
state advances a left NFA forward and a right NFA backward only when their
next characters agree.  This is deliberately different from a complete
sentence-pair sweep and from a word-level semordnilap bank: word boundaries
may cross on either side, and the two POS shapes need not be equal.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "tools" / "polaris" / "payload" / "brown.json.gz"
VOCAB = ROOT / "tools" / "polaris" / "payload" / "vocab30k.txt"
OUT = ROOT / "runs" / "packed-bidirectional-scene-shapes-20260928.json"


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1])
                     for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}


class NFA:
    def __init__(self):
        self.count = 1
        self.start = 0
        self.finish = 0
        self.edges: list[tuple[int, int, str, str, int]] = []
        self.epsilon: dict[int, list[int]] = defaultdict(list)

    def new(self) -> int:
        node = self.count
        self.count += 1
        return node

    def slot(self, alternatives: list[str], slot_index: int) -> None:
        before, after = self.finish, self.new()
        for phrase in alternatives:
            tape = norm(phrase)
            if not tape:
                self.epsilon[before].append(after)
                continue
            current = before
            for offset, char in enumerate(tape):
                following = after if offset == len(tape) - 1 else self.new()
                self.edges.append((current, following, char,
                                   phrase if offset == 0 else "", slot_index))
                current = following
        self.finish = after


def compile_shape(shape: tuple[str, ...], by_tag: dict[str, list[str]]) -> NFA:
    nfa = NFA()
    for i, tag in enumerate(shape):
        nfa.slot(by_tag[tag], i)
    return nfa


def load_lexicon(limit: int = 48) -> dict[str, list[str]]:
    import gzip
    blob = json.loads(gzip.open(PAYLOAD, "rt").read())
    tags = {word.casefold(): set(values) for word, values in blob["table"].items()}
    # The frozen vocabulary is frequency ordered.  Keeping only its common
    # prefix removes Brown-tagged abbreviations and dialect fragments (for
    # example ``mah``) before they can masquerade as prose.  This is a lexical
    # admissibility screen, not a readability score.
    order = [re.sub(r"[^a-z]", "", row.casefold())
             for row in VOCAB.read_text().splitlines()[:5000]]
    wanted = {"DET", "ADJ", "NOUN", "VERB", "PRON", "ADP", "ADV", "NUM"}
    by_tag: dict[str, list[str]] = {tag: [] for tag in wanted}
    seen: set[str] = set()
    for word in order:
        if not word.isalpha() or len(word) < 2 or word in seen:
            continue
        brown_tags = tags.get(word, set())
        if not brown_tags:
            continue
        seen.add(word)
        for tag in wanted & brown_tags:
            if len(by_tag[tag]) < limit:
                by_tag[tag].append(word)
        if all(len(by_tag[tag]) >= limit for tag in wanted):
            break
    return by_tag


def reverse_word_mirror(left: list[str], right: list[str]) -> bool:
    lt = [norm(word) for word in left]
    rt = [norm(word) for word in right]
    return bool(lt) and lt == [word[::-1] for word in reversed(rt)]


def intersect(left: NFA, right: NFA, max_states: int = 120_000) -> dict[str, object]:
    @lru_cache(None)
    def fwd_closure(node: int) -> frozenset[int]:
        reached = {node}
        for nxt in left.epsilon[node]:
            reached.update(fwd_closure(nxt))
        return frozenset(reached)

    reverse_epsilon: dict[int, list[int]] = defaultdict(list)
    for src, dsts in right.epsilon.items():
        for dst in dsts:
            reverse_epsilon[dst].append(src)

    @lru_cache(None)
    def back_closure(node: int) -> frozenset[int]:
        reached = {node}
        for prev in reverse_epsilon[node]:
            reached.update(back_closure(prev))
        return frozenset(reached)

    fwd_edges: dict[int, list[int]] = defaultdict(list)
    back_edges: dict[int, list[int]] = defaultdict(list)
    for edge_id, (src, dst, _char, _word, _slot) in enumerate(left.edges):
        fwd_edges[src].append(edge_id)
    for edge_id, (src, dst, _char, _word, _slot) in enumerate(right.edges):
        back_edges[dst].append(edge_id)

    @lru_cache(None)
    def outgoing(node: int) -> dict[str, tuple[int, ...]]:
        result: dict[str, list[int]] = defaultdict(list)
        for src in fwd_closure(node):
            for edge_id in fwd_edges[src]:
                result[left.edges[edge_id][2]].append(edge_id)
        return {char: tuple(ids) for char, ids in result.items()}

    @lru_cache(None)
    def incoming(node: int) -> dict[str, tuple[int, ...]]:
        result: dict[str, list[int]] = defaultdict(list)
        for dst in back_closure(node):
            for edge_id in back_edges[dst]:
                result[right.edges[edge_id][2]].append(edge_id)
        return {char: tuple(ids) for char, ids in result.items()}

    @lru_cache(None)
    def reachable_left(node: int) -> frozenset[int]:
        reached = {node}
        for src in fwd_closure(node):
            for edge_id in fwd_edges[src]:
                reached.update(reachable_left(left.edges[edge_id][1]))
        return frozenset(reached)

    @lru_cache(None)
    def can_finish_left(node: int) -> bool:
        return left.finish in reachable_left(node)

    @lru_cache(None)
    def can_start_right(node: int) -> bool:
        return right.start in back_closure(node)

    queue = deque([(left.start, right.finish, (), ())])
    seen: set[tuple[int, int]] = set()
    accepting: list[dict[str, object]] = []
    dead: list[dict[str, object]] = []
    transitions = 0
    while queue and len(seen) < max_states:
        left_node, right_node, left_path, right_path = queue.popleft()
        key = (left_node, right_node)
        if key in seen:
            continue
        seen.add(key)
        if can_finish_left(left_node) and can_start_right(right_node):
            left_words = [left.edges[i][3] for i in left_path if left.edges[i][3]]
            right_forward_ids = list(reversed(right_path))
            right_words = [right.edges[i][3] for i in right_forward_ids if right.edges[i][3]]
            rendered = " ".join(left_words + right_words)
            checked = audit(rendered)
            if checked["two_pointer_exact"]:
                accepting.append({"rendered": rendered, "audit": checked,
                                  "left_words": left_words, "right_words": right_words,
                                  "word_order_mirror": reverse_word_mirror(left_words, right_words)})
            continue
        out, inc = outgoing(left_node), incoming(right_node)
        common = sorted(out.keys() & inc.keys())
        if not common:
            dead.append({"matched_pairs": len(left_path), "left_node": left_node,
                         "right_node": right_node, "left_next": sorted(out),
                         "right_next": sorted(inc)})
            dead.sort(key=lambda row: -row["matched_pairs"])
            del dead[32:]
            continue
        for char in common:
            for left_edge in out[char]:
                next_left = left.edges[left_edge][1]
                for right_edge in inc[char]:
                    next_right = right.edges[right_edge][0]
                    if not can_finish_left(next_left) or not can_start_right(next_right):
                        # It may still need more characters, but reject only
                        # impossible graph states, never by numeric node order.
                        if not reachable_left(next_left) or not back_closure(next_right):
                            continue
                    queue.append((next_left, next_right,
                                 left_path + (left_edge,), right_path + (right_edge,)))
                    transitions += 1
    return {"states": len(seen), "transitions": transitions,
            "cap_reached": bool(queue), "accepting": accepting,
            "dead_frontiers": dead, "left_states": left.count,
            "right_states": right.count}


SHAPES = (
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"),
)


def run() -> dict[str, object]:
    by_tag = load_lexicon()
    rows = []
    searches = []
    for left_shape in SHAPES:
        for right_shape in SHAPES:
            left_nfa = compile_shape(left_shape, by_tag)
            right_nfa = compile_shape(right_shape, by_tag)
            result = intersect(left_nfa, right_nfa)
            searches.append({"left_shape": left_shape, "right_shape": right_shape,
                             "states": result["states"], "transitions": result["transitions"],
                             "cap_reached": result["cap_reached"],
                             "dead_frontiers": result["dead_frontiers"][:4]})
            for row in result["accepting"]:
                if row["word_order_mirror"]:
                    continue
                rows.append({**row, "left_shape": left_shape, "right_shape": right_shape,
                             "provenance": {
                                 "construction": "packed bidirectional POS-shape intersection",
                                 "fresh_lexical_combinations": True,
                                 "complete_sentence_enumeration": False,
                                 "finished_tape_reversal": False,
                                 "word_order_mirror": False,
                                 "catalogue_text": False,
                                 "per_candidate_rlaif": False,
                                 "reader_certified": False,
                             }})
    unique = {row["audit"]["sha256_forward"]: row for row in rows}
    return {"experiment_id": "packed-bidirectional-scene-shapes-20260928",
            "method": "packed live character intersection of independent POS sentence shapes",
            "shapes": [list(shape) for shape in SHAPES],
            "lexicon_sizes": {tag: len(words) for tag, words in by_tag.items()},
            "searches": searches, "exact_non_word_mirror": list(unique.values()),
            "controls": [
                {"kind": "intact_prose_control", "rendered": "The pilot maps the chart near the harbor.",
                 "audit": audit("The pilot maps the chart near the harbor.")},
                {"kind": "incumbent_exact_control",
                 "rendered": "An aide rips nine memos; some men inspire Diana.",
                 "audit": audit("An aide rips nine memos; some men inspire Diana.")},
            ],
            "reader_gate": "closed; no blinded human ratings collected",
            "provenance": {"generator_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
                "independent_audits": ["two-pointer", "forward/reverse SHA-256"]}}


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"searches": len(payload["searches"]),
                      "exact_non_word_mirror": len(payload["exact_non_word_mirror"])}, sort_keys=True))
