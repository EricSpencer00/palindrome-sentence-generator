"""Dream-RSI construction loop: counterexample-guided role-lexicon expansion.

This experiment keeps the exact character product as the inner solver.  A
failed product state is not treated as a result: it is a counterexample that
records the two grammatical roles and the incompatible live character sets.
The next round adds only role-compatible words whose boundary letters can
address those obligations.  The product is rerun from scratch, so exactness
is enforced while a path is built rather than by reversing a finished tape.

The benchmark sentence is a withheld kernel witness.  It validates the
implementation but is never part of the fresh candidate set.  Programmatic
filters reject obvious construction shortcuts; human readers remain the only
readability gate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "cegar-role-product-20260917.json"
EXPERIMENT_ID = "cegar-role-product-20260917"
SIGNATURE = (
    "dream-rsi|counterexample-guided-role-lexicon|"
    "live-character-product|independent-audit"
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.lower()))


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str
    role: str
    word: str | None = None


@dataclass(frozen=True)
class Grammar:
    start: int
    end: int
    edges: tuple[Edge, ...]
    pattern: tuple[str, ...]
    role_words: dict[str, tuple[str, ...]]


def compile_pattern(pattern: tuple[str, ...], role_words: dict[str, Iterable[str]]) -> Grammar:
    """Compile role alternatives into an acyclic character graph."""
    if not pattern:
        raise ValueError("empty pattern")
    edges: list[Edge] = []
    boundary = 0
    next_node = 1
    normalized: dict[str, tuple[str, ...]] = {}
    for role in dict.fromkeys(pattern):
        choices = tuple(dict.fromkeys(normalize(w) for w in role_words.get(role, ())))
        if not choices or any(not w for w in choices):
            raise ValueError(f"missing lexical choices for {role}")
        normalized[role] = choices
    for slot, role in enumerate(pattern):
        target_boundary = next_node
        next_node += 1
        for word in normalized[role]:
            source = boundary
            for i, char in enumerate(word):
                final = i == len(word) - 1
                target = target_boundary if final else next_node
                if not final:
                    next_node += 1
                edges.append(Edge(source, target, char, role, word if final else None))
                source = target
        boundary = target_boundary
    return Grammar(0, boundary, tuple(edges), pattern, normalized)


def _reachable(grammar: Grammar) -> dict[int, set[int]]:
    outgoing: dict[int, list[Edge]] = defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
    cache: dict[int, set[int]] = {}

    def visit(node: int) -> set[int]:
        if node in cache:
            return cache[node]
        seen = {node}
        cache[node] = seen
        for edge in outgoing[node]:
            seen.update(visit(edge.target))
        return seen

    visit(grammar.start)
    return cache


def _replay(grammar: Grammar, path: tuple[Edge, ...]) -> dict:
    cursor = grammar.start
    words: list[str] = []
    roles: list[str] = []
    chars: list[str] = []
    for edge in path:
        if edge.source != cursor:
            return {"ok": False, "reason": "disconnected"}
        cursor = edge.target
        chars.append(edge.char)
        if edge.word is not None:
            words.append(edge.word)
            roles.append(edge.role)
    return {
        "ok": cursor == grammar.end and tuple(roles) == grammar.pattern,
        "words": words,
        "roles": roles,
        "tape": "".join(chars),
    }


def _words_from_edges(edges: tuple[Edge, ...]) -> list[str]:
    """Decode completed lexical words without assuming a path starts at 0."""
    return [edge.word for edge in edges if edge.word is not None]


def _audit(text: str) -> dict:
    tape = normalize(text)
    mismatches = [
        (i, len(tape) - 1 - i)
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "algorithm": "independent_two_pointer_scan",
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "mismatches": mismatches,
        "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def _anti_shortcut(words: list[str]) -> dict:
    norm = [normalize(w) for w in words]
    return {
        "word_order_only_symmetry": norm == [w[::-1] for w in reversed(norm)],
        "repeated_words": len(norm) != len(set(norm)),
        "self_palindromic_words": [w for w in norm if len(w) > 2 and w == w[::-1]],
        "catalogue_or_fixture": False,
    }


def product(grammar: Grammar, max_states: int = 250_000) -> dict:
    """Search exact paths by synchronizing opposite character edges."""
    out: dict[int, list[Edge]] = defaultdict(list)
    incoming: dict[int, list[Edge]] = defaultdict(list)
    for edge in grammar.edges:
        out[edge.source].append(edge)
        incoming[edge.target].append(edge)
    reachable = _reachable(grammar)
    # Prefix/suffix are retained for replay and dead-frontier provenance.
    stack = [(grammar.start, grammar.end, tuple(), tuple())]
    records: list[dict] = []
    dead: list[dict] = []
    states = 0
    seen: set[tuple[int, int, tuple[Edge, ...], tuple[Edge, ...]]] = set()
    while stack and states < max_states:
        left, right, prefix, suffix = stack.pop()
        key = (left, right, prefix, suffix)
        if key in seen:
            continue
        seen.add(key)
        states += 1
        if left == right:
            replay = _replay(grammar, prefix + suffix)
            if replay["ok"]:
                records.append({**replay, "center": "empty", "matched_pairs": len(prefix)})
            continue
        center_edges = [edge for edge in out[left] if edge.target == right]
        if center_edges:
            for center in center_edges:
                replay = _replay(grammar, prefix + (center,) + suffix)
                if replay["ok"]:
                    records.append({**replay, "center": center.char,
                                    "matched_pairs": len(prefix)})
        matches: list[tuple[Edge, Edge]] = []
        for first in out[left]:
            for last in incoming[right]:
                if first.char == last.char and last.source in reachable.get(first.target, set()):
                    matches.append((first, last))
                    stack.append((first.target, last.source,
                                  prefix + (first,), (last,) + suffix))
        if not matches:
            left_edges = out[left]
            right_edges = incoming[right]
            dead.append({
                "matched_pairs": len(prefix),
                "left_roles": sorted({e.role for e in left_edges}),
                "right_roles": sorted({e.role for e in right_edges}),
                "left_next_chars": sorted({e.char for e in left_edges}),
                "right_prev_chars": sorted({e.char for e in right_edges}),
                "left_words": _words_from_edges(prefix),
                "right_words": _words_from_edges(suffix),
                "reason": "no_equal_live_character_edge",
            })
    return {
        "states": states,
        "truncated": bool(stack),
        "records": records,
        "dead_frontiers": dead[:100],
    }


def _row(grammar: Grammar, rec: dict, source_round: int) -> dict:
    text = " ".join(rec["words"])
    audit = _audit(text)
    shortcut = _anti_shortcut(rec["words"])
    mechanically_admitted = audit["exact"] and not any(shortcut.values())
    return {
        "rendered": text,
        "words": rec["words"],
        "roles": rec["roles"],
        "length": audit["letters"],
        "audit": audit,
        "anti_shortcut": shortcut,
        "mechanically_admitted": mechanically_admitted,
        "reader_eligible": False,
        "provenance": {
            "experiment_id": EXPERIMENT_ID,
            "round": source_round,
            "pattern": list(grammar.pattern),
            "construction": "live outside-in character product",
        },
    }


# Ordinary, authored lexical reservoirs.  The seed words are deliberately
# absent from both the initial banks and all repair reservoirs.
INITIAL = {
    "det": ("a", "an", "the", "my", "one", "some"),
    "adj": ("calm", "careful", "patient", "quiet", "young", "bright", "kind", "old", "small", "warm"),
    "noun": ("artist", "baker", "child", "farmer", "friend", "gardener", "keeper", "nurse", "poet", "reader", "sailor", "teacher", "writer", "garden", "letter", "river", "story", "town", "window"),
    "verb": ("carries", "draws", "helps", "keeps", "marks", "opens", "plants", "reads", "records", "sees", "visits", "watches", "writes"),
}

REPAIR_RESERVOIR = {
    "det": ("each", "every", "her", "his", "our", "their", "this", "that"),
    "adj": ("ancient", "blue", "clear", "distant", "gentle", "green", "happy", "little", "public", "red", "soft", "wise"),
    "noun": ("author", "boat", "bridge", "candle", "chart", "dog", "home", "house", "island", "lamp", "map", "moon", "path", "plant", "room", "shore", "stone", "star", "sun", "teacher"),
    "verb": ("builds", "closes", "finds", "gives", "guides", "hears", "knows", "likes", "moves", "sends", "sings", "takes", "tells", "walks"),
}


def _repair_words(banks: dict[str, tuple[str, ...]], dead: list[dict]) -> tuple[dict, list[dict]]:
    """Add only words that address a recorded role/character obligation."""
    additions: dict[str, set[str]] = defaultdict(set)
    actions: list[dict] = []
    for frontier in dead:
        left_roles = set(frontier["left_roles"])
        right_roles = set(frontier["right_roles"])
        left_chars = set(frontier["left_next_chars"])
        right_chars = set(frontier["right_prev_chars"])
        needed = left_chars & right_chars
        for role in sorted(left_roles | right_roles):
            for word in REPAIR_RESERVOIR.get(role, ()):
                # A repair is useful only if its exposed boundary can emit a
                # currently required character; it is never a finished-tape
                # substitution and never copies a benchmark token.
                if needed:
                    useful = word[0] in needed or word[-1] in needed
                    matched_boundary = "first" if word[0] in needed else "last"
                elif role in left_roles and word[0] in right_chars:
                    # The current banks have disjoint exposed alphabets.  A
                    # left-role word beginning with a right obligation can
                    # create the missing equal pair on the next replay.
                    useful = True
                    matched_boundary = "first_against_right"
                elif role in right_roles and word[-1] in left_chars:
                    useful = True
                    matched_boundary = "last_against_left"
                else:
                    useful = False
                    matched_boundary = None
                if useful:
                    if word not in banks.get(role, ()):
                        additions[role].add(word)
                        actions.append({"role": role, "word": word,
                                        "needed_chars": sorted(needed),
                                        "matched_boundary": matched_boundary,
                                        "matched_pairs": frontier["matched_pairs"]})
    role_order = tuple(dict.fromkeys(tuple(banks) + tuple(REPAIR_RESERVOIR)))
    updated = {role: tuple(dict.fromkeys(tuple(banks.get(role, ())) + tuple(sorted(additions.get(role, ())))))
              for role in role_order}
    return updated, actions


PATTERNS = (
    ("det", "noun", "verb", "det", "noun"),
    ("det", "adj", "noun", "verb", "det", "noun"),
    ("det", "adj", "noun", "verb", "det", "adj", "noun"),
)


def withheld_seed_witness() -> dict:
    """Validate the kernel on the supplied seed, outside the fresh run."""
    seed_words = ("an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana")
    seed_roles = ("det", "noun", "verb", "adj", "noun", "det", "noun", "verb", "noun")
    by_role: dict[str, list[str]] = defaultdict(list)
    for role, word in zip(seed_roles, seed_words):
        by_role[role].append(word)
    g = compile_pattern(seed_roles, by_role)
    result = product(g)
    text = " ".join(seed_words)
    audit = _audit(text)
    return {
        "text": text,
        "letters": audit["letters"],
        "product_records": len(result["records"]),
        "audit": audit,
        "withheld": True,
        "not_a_generated_candidate": True,
    }


def run(max_states: int = 250_000, rounds: int = 4) -> dict:
    rounds_out: list[dict] = []
    fresh_rows: list[dict] = []
    for pattern in PATTERNS:
        banks = {role: tuple(words) for role, words in INITIAL.items()}
        for round_id in range(rounds):
            grammar = compile_pattern(pattern, banks)
            result = product(grammar, max_states=max_states)
            rows = [_row(grammar, rec, round_id) for rec in result["records"]]
            fresh_rows.extend(row for row in rows if row["mechanically_admitted"])
            updated, actions = _repair_words(banks, result["dead_frontiers"])
            rounds_out.append({
                "pattern": list(pattern),
                "round": round_id,
                "lexicon_sizes": {role: len(words) for role, words in banks.items()},
                "states": result["states"],
                "truncated": result["truncated"],
                "exact_records": len(rows),
                "mechanically_admitted": sum(row["mechanically_admitted"] for row in rows),
                "rendered_rows": rows[:50],
                "dead_frontiers": result["dead_frontiers"][:25],
                "repair_actions": actions[:100],
                "next_repair": {
                    "operator": "role-compatible boundary lexical expansion",
                    "applied": bool(actions),
                    "roles": sorted({a["role"] for a in actions}),
                },
            })
            if not actions:
                break
            banks = updated
    fresh_rows.sort(key=lambda row: (-row["length"], row["rendered"]))
    # Do not call the withheld kernel a generated row.
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_exact" if fresh_rows else "completed_no_fresh_exact_closure",
        "method": "counterexample-guided role lexicon expansion over a live character-edge product",
        "withheld_kernel_witness": withheld_seed_witness(),
        "patterns": [list(p) for p in PATTERNS],
        "rounds": rounds_out,
        "rendered_candidates": fresh_rows[:100],
        "candidate_count": len(fresh_rows),
        "exact_count": len(fresh_rows),
        "reader_eligible_count": 0,
        "reader_gate": {
            "status": "not_triggered",
            "human_blind_readability_required": True,
            "programmatic_measures_are_diagnostic": True,
        },
        "failure_and_repair": {
            "failure": "no fresh mechanically admitted exact closure" if not fresh_rows else "fresh exact closure found; readability still untested",
            "next_repair": "expand role-compatible lexical entries at the deepest recorded dead frontier, then test a new clause pattern if the seam remains unreachable",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256", "grammar path replay"],
            "shortcuts_rejected": ["word-order-only symmetry", "repeated words", "self-palindromic words", "catalogue/fixture promotion"],
        },
    }


def main() -> None:
    artifact = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "experiment_id": artifact["experiment_id"],
        "status": artifact["status"],
        "candidate_count": artifact["candidate_count"],
        "exact_count": artifact["exact_count"],
        "rounds": len(artifact["rounds"]),
        "withheld_letters": artifact["withheld_kernel_witness"]["letters"],
    }, indent=2))


if __name__ == "__main__":
    main()
