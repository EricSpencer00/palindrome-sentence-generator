"""Replay the staggered ``an eraser`` / ``an arena`` endpoint from character 0."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re


ID = "fresh-endpoint-terminal-clause-trie-20261002"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / f"{ID}.json"
FUNCTION_WORDS = frozenset("a an the it then later she".split())
REMOTE_ORIGIN = {
    "source_path": "/home/eric/pal-fresh-run/experiments/fresh_endpoint_terminal_clause_trie_20260922.py",
    "source_sha256": "a6f44b33ed39316493d7bd8b8b47d8ca4daef1fc4ebe0a4dd078f486767fdf33",
    "result_path": "/home/eric/pal-fresh-run/runs/fresh-endpoint-terminal-clause-trie-20260922.json",
    "result_sha256": "8aad9a5d5b5485170a4eaba9c9597b20833a5d6ee9bc75e190df0b369807e535",
    "log_sha256": "db9f647b289d791d0f03c3fdd82e0c3b189313167d9e5dab04c78d83101f9b1c",
    "audit_sha256": "bc1dcd8a043e731169cf5ef541c94905b8fdfe4c49265f9e44b20d0dcbb0851f",
}


@dataclass(frozen=True)
class Event:
    actor: str
    number: str
    predicate: str
    agreement: str
    valency: str
    patient: str | None
    relation: str


@dataclass(frozen=True)
class Discourse:
    discourse_id: str
    sentences: tuple[tuple[str, ...], ...]
    events: tuple[Event, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(word for sentence in self.sentences for word in sentence)

    @property
    def rendered(self) -> str:
        return " ".join(" ".join(sentence).capitalize() + "." for sentence in self.sentences)


LEFT = (
    Discourse(
        "eraser-removes-mark",
        (("an", "eraser", "removes", "a", "mark"),
         ("it", "leaves", "the", "paper", "clean")),
        (Event("eraser", "sg", "removes", "sg", "transitive", "mark", "cause"),
         Event("eraser", "sg", "leaves", "sg", "resultative", "paper", "result_of:0")),
    ),
    Discourse(
        "eraser-erases-line",
        (("an", "eraser", "erases", "a", "line"),
         ("it", "leaves", "the", "page", "clear")),
        (Event("eraser", "sg", "erases", "sg", "transitive", "line", "cause"),
         Event("eraser", "sg", "leaves", "sg", "resultative", "page", "result_of:0")),
    ),
)
RIGHT = (
    Discourse(
        "crew-prepares-arena",
        (("a", "crew", "clears", "a", "gate"),
         ("then", "it", "prepares", "an", "arena")),
        (Event("crew", "sg", "clears", "sg", "transitive", "gate", "preparation"),
         Event("crew", "sg", "prepares", "sg", "transitive", "arena", "after:0")),
    ),
    Discourse(
        "team-enters-arena",
        (("a", "team", "trains", "nearby"),
         ("then", "it", "enters", "an", "arena")),
        (Event("team", "sg", "trains", "sg", "intransitive", None, "preparation"),
         Event("team", "sg", "enters", "sg", "transitive", "arena", "after:0")),
    ),
)


@dataclass
class TrieNode:
    edges: dict[str, "TrieNode"] = field(default_factory=dict)
    terminals: set[str] = field(default_factory=set)


def audit_discourse(discourse: Discourse) -> dict[str, bool]:
    return {
        "two_events": len(discourse.events) == 2,
        "event_continuity": discourse.events[0].actor == discourse.events[1].actor,
        "singular_agreement": all(
            event.number == event.agreement == "sg" for event in discourse.events
        ),
        "valency_satisfied": all(
            (event.patient is not None) == (event.valency in {"transitive", "resultative"})
            for event in discourse.events
        ),
        "common_lowercase_words": all(
            word.isascii() and word.isalpha() and word.islower() for word in discourse.words
        ),
        "no_self_palindromic_content_word": all(
            word in FUNCTION_WORDS or word != word[::-1] for word in discourse.words
        ),
    }


def build_trie(discourses: tuple[Discourse, ...], *, reverse: bool) -> TrieNode:
    root = TrieNode()
    for discourse in discourses:
        checks = audit_discourse(discourse)
        if not all(checks.values()):
            raise AssertionError((discourse.discourse_id, checks))
        sequence = tuple(reversed(discourse.words)) if reverse else discourse.words
        node = root
        for word in sequence:
            node = node.edges.setdefault(word, TrieNode())
        node.terminals.add(discourse.discourse_id)
    return root


def compare(owner: str, residual: str, side: str, exposed: str) -> tuple[str, str] | None:
    if not residual:
        return side, exposed
    common = min(len(residual), len(exposed))
    if residual[:common] != exposed[:common]:
        return None
    if len(residual) > len(exposed):
        return owner, residual[common:]
    if len(exposed) > len(residual):
        return side, exposed[common:]
    return "", ""


def run() -> dict:
    left_node = build_trie(LEFT, reverse=False)
    right_node = build_trie(RIGHT, reverse=True)
    owner = residual = ""
    left_cursor = right_cursor = 0
    left_boundaries: set[int] = set()
    right_boundaries: set[int] = set()
    trace = []

    def emit(side: str, word: str) -> bool:
        nonlocal left_node, right_node, owner, residual, left_cursor, right_cursor
        node = left_node if side == "left" else right_node
        if word not in node.edges:
            raise AssertionError((side, word, sorted(node.edges)))
        exposed = word if side == "left" else word[::-1]
        before = {"owner": owner, "residual": residual}
        outcome = compare(owner, residual, side, exposed)
        if outcome is None:
            trace.append({
                "side": side, "word": word, "exposed": exposed,
                "accepted": False, "matched_cursor": min(left_cursor, right_cursor),
                "before": before,
                "first_conflict": [residual[0], exposed[0]],
            })
            return False
        next_owner, next_residual = outcome
        if side == "left":
            left_cursor += len(word); left_boundaries.add(left_cursor)
            left_node = node.edges[word]
            boundary, opposite = left_cursor, right_boundaries
        else:
            right_cursor += len(word); right_boundaries.add(right_cursor)
            right_node = node.edges[word]
            boundary, opposite = right_cursor, left_boundaries
        if boundary in opposite and next_residual:
            raise AssertionError(f"forbidden complementary boundary at {boundary}")
        owner, residual = next_owner, next_residual
        trace.append({
            "side": side, "word": word, "exposed": exposed, "accepted": True,
            "before": before, "after": {"owner": owner, "residual": residual},
            "left_boundaries": sorted(left_boundaries),
            "right_exposed_boundaries": sorted(right_boundaries),
        })
        return True

    # Forced endpoint replay from normalized cursor zero.
    assert emit("left", "an")
    assert emit("right", "arena")
    assert emit("left", "eraser")
    seed = {
        "matched_prefix": "anera",
        "matched_letters": min(left_cursor, right_cursor),
        "owner": owner, "residual": residual,
        "left_boundaries": sorted(left_boundaries),
        "right_exposed_boundaries": sorted(right_boundaries),
        "complementary_boundaries": sorted(left_boundaries & right_boundaries),
        "next_left_predicates": sorted(left_node.edges),
        "next_right_required_tokens": sorted(right_node.edges),
    }
    assert seed["matched_letters"] == 5 and residual == "ser"
    assert not seed["complementary_boundaries"]

    attempts = []
    for word in sorted(right_node.edges):
        accepted = emit("right", word)
        attempts.append({"surface": word, "exposed": word[::-1], "accepted": accepted})
        assert not accepted

    return {
        "experiment_id": ID,
        "method": "endpoint-conditioned typed terminal-clause trie",
        "grammar": {
            "left": [discourse.rendered for discourse in LEFT],
            "right": [discourse.rendered for discourse in RIGHT],
            "audits": {
                discourse.discourse_id: audit_discourse(discourse)
                for discourse in (*LEFT, *RIGHT)
            },
        },
        "seed": seed,
        "stats": {
            "left_discourse_paths": len(LEFT), "right_discourse_paths": len(RIGHT),
            "accepted_token_transitions": sum(row["accepted"] for row in trace),
            "rejected_token_transitions": sum(not row["accepted"] for row in trace),
            "predicate_unification_attempts": 0,
            "exact_rendered_candidates": 0, "cap_reached": False,
        },
        "trace": trace,
        "attempts_after_seed": attempts,
        "exact_candidates": [],
        "obstruction": {
            "kind": "mandatory_determiner_precedes_adjacent_predicate_in_reverse_exposure",
            "matched_cursor_zero_based": 5,
            "left_owner_residual": "ser",
            "right_next_surface_token": "an",
            "right_next_exposed_characters": "na",
            "first_conflict": {"left": "s", "right": "n"},
            "proof": (
                "Every grammatical right path ends with the NP 'an arena'. After "
                "reverse(arena) consumes 'anera', left debt is 'ser'. The right "
                "grammar must next expose reverse(an)='na', so s != n at cursor 5. "
                "The predicate is a later trie edge and cannot be reordered."
            ),
        },
        "provenance": {
            "remote_origin": REMOTE_ORIGIN,
            "repo_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sol_only": True, "host_of_original_run": "hst-bench",
            "search_started_at_first_character": True,
            "proper_names": False, "catalogue_text": False,
            "finished_tape_reversal": False, "posthoc_repair": False,
            "duplicate_sweep": False, "complementary_boundary_mask_live": True,
            "lane_closed": True,
        },
        "disposition": (
            "Stop this fixed-determiner lane at the exact s!=n obstruction; do not "
            "run another paragraph or determiner variant."
        ),
    }


def main() -> None:
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
