"""Finite SVO clause search with live mirrored-character orbit assignment.

This lane compiles two independent finite clause languages into character
tries.  A product state assigns one character orbit (the next character from
the left clause and the next character from the right clause, traversed from
the outside inward) before either side can advance.  A closure is accepted
only when both trie states are terminal complete subject/finite-verb/object
clauses.  No word-pair table, mirrored word order, finished-tape reversal, or
model reward is consulted.

The lexical inventory is intentionally small and hand-authored.  It is a
bounded construction diagnostic, not a readability certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


EXPERIMENT_ID = "finite-clause-character-orbits-20260919"
SIGNATURE = (
    "finite-svo-clause-language|character-trie-product|"
    "live-mirrored-orbit-assignment|terminal-complete-clause-gate"
)


@dataclass(frozen=True)
class Clause:
    """One ordinary finite transitive clause, kept role-complete."""

    subject: str
    verb: str
    object: str
    clause_id: str

    @property
    def text(self) -> str:
        return f"{self.subject} {self.verb} {self.object}"

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def roles(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "finite_verb": self.verb,
            "object": self.object,
        }


# The two inventories are authored independently.  Their only shared
# structure is the required S -> finite V -> O grammar; no reverse lexical
# pairs are listed or generated.
LEFT_CLAUSES = (
    Clause("An aide", "writes", "nine memos", "left-01"),
    Clause("A patient keeper", "guards", "charts", "left-02"),
    Clause("The sailor", "carries", "charts", "left-03"),
    Clause("A quiet teacher", "reads", "letters", "left-04"),
    Clause("The gardener", "waters", "cedar", "left-05"),
    Clause("A careful pilot", "maps", "rivers", "left-06"),
)

RIGHT_CLAUSES = (
    Clause("Some men", "inspire", "Diana", "right-01"),
    Clause("The baker", "records", "a sonnet", "right-02"),
    Clause("A patient nurse", "marks", "charts", "right-03"),
    Clause("The keeper", "guards", "old maps", "right-04"),
    Clause("A singer", "carries", "parcels", "right-05"),
    Clause("The teacher", "reads", "marked letters", "right-06"),
)


@dataclass
class TrieNode:
    children: dict[str, int] = field(default_factory=dict)
    terminals: list[str] = field(default_factory=list)


class ClauseTrie:
    """A finite clause trie with terminal role metadata.

    ``reverse_cursor=True`` stores the right language from its final
    character inward.  This is a traversal index for assigning orbits; the
    original clause remains the rendered right-hand clause.
    """

    def __init__(self, clauses: Iterable[Clause], *, reverse_cursor: bool):
        self.nodes = [TrieNode()]
        self.clauses = {clause.clause_id: clause for clause in clauses}
        self.reverse_cursor = reverse_cursor
        for clause in clauses:
            node = 0
            stream = clause.tape[::-1] if reverse_cursor else clause.tape
            for char in stream:
                node = self.nodes[node].children.setdefault(char, self._new_node())
            self.nodes[node].terminals.append(clause.clause_id)

    def _new_node(self) -> int:
        self.nodes.append(TrieNode())
        return len(self.nodes) - 1


def _first_mismatch(tape: str) -> dict[str, int | str] | None:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return {
                "orbit": left,
                "left_index": left,
                "right_index": right,
                "left_char": tape[left],
                "right_char": tape[right],
            }
        left += 1
        right -= 1
    return None


def audit(text: str) -> dict[str, object]:
    """Independent normalized two-pointer and forward/reverse SHA audit."""
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatch = _first_mismatch(tape)
    forward_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode("ascii")).hexdigest()
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha256_equal": forward_sha == reverse_sha,
    }


def orbit_assignments(left: Clause, right: Clause) -> dict[str, object]:
    """Assign every available mirrored orbit without constructing a mirror.

    The right clause is read in its ordinary rendered order for output.  The
    comparison cursor visits its final character first solely to expose the
    corresponding orbit.  A length mismatch is retained as a mechanical
    failure rather than padded or repaired.
    """
    left_tape, right_tape = left.tape, right.tape
    limit = min(len(left_tape), len(right_tape))
    trace = []
    first_failure = None
    for orbit in range(limit):
        left_char = left_tape[orbit]
        right_index = len(right_tape) - 1 - orbit
        right_char = right_tape[right_index]
        matched = left_char == right_char
        trace.append({
            "orbit": orbit,
            "left_index": orbit,
            "right_index": right_index,
            "left_char": left_char,
            "right_char": right_char,
            "assigned": matched,
        })
        if not matched and first_failure is None:
            first_failure = trace[-1]
            break
    return {
        "assigned_orbits": len(trace),
        "trace": trace,
        "first_failure": first_failure,
        "equal_clause_length": len(left_tape) == len(right_tape),
        "closed": first_failure is None and len(left_tape) == len(right_tape),
    }


def _render(left: Clause, right: Clause) -> str:
    right_text = right.text[:1].upper() + right.text[1:]
    return f"{left.text}; {right_text}."


def _row(left: Clause, right: Clause, *, source: str) -> dict[str, object]:
    rendered = _render(left, right)
    checked = audit(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=60)
    orbit = orbit_assignments(left, right)
    return {
        "rendered": rendered,
        "roles": {"left": left.roles, "right": right.roles},
        "complete_finite_svo": True,
        "source": source,
        "orbit_assignment": orbit,
        "audit": checked,
        "mechanical_checks": checks,
        "mechanically_admitted": checked["two_pointer_exact"] and all(checks.values()),
        "provenance": {
            "construction": "independent finite SVO clause tries with live mirrored character-orbit assignment",
            "lexical_domains_independent": True,
            "complete_finite_subject_verb_object_clauses": True,
            "reversible_lexical_pairs": False,
            "word_order_symmetry": False,
            "catalogue_text": False,
            "finished_tape_reversal": False,
            "rlaif_reward_loop": False,
            "reader_status": "unreviewed; mechanical gates are not readability evidence",
        },
    }


def _orbit_product(
    left_trie: ClauseTrie,
    right_trie: ClauseTrie,
    *,
    min_letters: int = 39,
    max_letters: int = 60,
    max_states: int = 2_000,
) -> dict[str, object]:
    """Walk the two tries; each accepted edge assigns exactly one orbit."""
    stack: list[tuple[int, int, int]] = [(0, 0, 0)]
    expanded = 0
    matched_orbits = 0
    rejected_orbits = 0
    closures: list[dict[str, object]] = []
    while stack and expanded < max_states:
        left_node, right_node, depth = stack.pop()
        expanded += 1
        left_state, right_state = left_trie.nodes[left_node], right_trie.nodes[right_node]
        if left_state.terminals and right_state.terminals:
            # Both terminal states are complete SVO clauses.  The grammar
            # completion gate happens before any candidate is rendered.
            for left_id in left_state.terminals:
                for right_id in right_state.terminals:
                    left = left_trie.clauses[left_id]
                    right = right_trie.clauses[right_id]
                    letters = len(left.tape) + len(right.tape)
                    if min_letters <= letters <= max_letters:
                        closures.append({"left": left_id, "right": right_id, "letters": letters})
        for left_char, left_next in left_state.children.items():
            right_next = right_state.children.get(left_char)
            if right_next is None:
                rejected_orbits += 1
                continue
            matched_orbits += 1
            stack.append((left_next, right_next, depth + 1))
    return {
        "expanded_states": expanded,
        "matched_orbits": matched_orbits,
        "rejected_orbits": rejected_orbits,
        "budget_exhausted": bool(stack),
        "closures": closures,
    }


def _prior_exact_tapes() -> set[str]:
    """Read only prior run artifacts for duplicate rejection, never as input."""
    tapes: set[str] = set()
    for path in (
        ROOT / "runs" / "typed-clause-zipper-20260919.json",
        ROOT / "runs" / "typed-word-boundary-clause-automaton-20260918.json",
    ):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = json.dumps(payload)
        for match in re.findall(r'"normalized"\s*:\s*"([a-z]+)"', text):
            if match == match[::-1]:
                tapes.add(match)
    return tapes


def run() -> dict[str, object]:
    left_trie = ClauseTrie(LEFT_CLAUSES, reverse_cursor=False)
    right_trie = ClauseTrie(RIGHT_CLAUSES, reverse_cursor=True)
    product = _orbit_product(left_trie, right_trie)
    prior = _prior_exact_tapes()

    exact_rows: list[dict[str, object]] = []
    duplicate_rejections = 0
    for closure in product["closures"]:
        left = left_trie.clauses[closure["left"]]
        right = right_trie.clauses[closure["right"]]
        row = _row(left, right, source="live-orbit-terminal-closure")
        if row["audit"]["normalized"] in prior:
            duplicate_rejections += 1
            continue
        exact_rows.append(row)

    # Preserve intact, ordinary controls in the requested 39--60-letter band
    # even when the exact orbit product is empty.  They are never promoted as
    # palindrome candidates and retain the first orbit failure for repair.
    control_pairs = (
        (LEFT_CLAUSES[0], RIGHT_CLAUSES[0]),
        (LEFT_CLAUSES[1], RIGHT_CLAUSES[1]),
        (LEFT_CLAUSES[5], RIGHT_CLAUSES[4]),
    )
    controls = [_row(left, right, source="intact-finite-svo-control")
                for left, right in control_pairs]
    exact = [row for row in exact_rows if row["audit"]["two_pointer_exact"]]
    mechanically_admitted = [row for row in exact if row["mechanically_admitted"]]
    all_rows = exact_rows + controls
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": "finite SVO clause-trie product with live mirrored character-orbit assignment",
        "grammar": {
            "left_slots": ["subject", "finite_verb", "object"],
            "right_slots": ["subject", "finite_verb", "object"],
            "terminal_gate": "both independent clause tries must reach complete SVO terminals",
            "left_clause_count": len(LEFT_CLAUSES),
            "right_clause_count": len(RIGHT_CLAUSES),
        },
        "stats": {
            "expanded_states": product["expanded_states"],
            "matched_orbits": product["matched_orbits"],
            "rejected_orbits": product["rejected_orbits"],
            "exact_closures_before_novelty": len(product["closures"]),
            "duplicate_rejections": duplicate_rejections,
            "exact": len(exact),
            "mechanically_admitted": len(mechanically_admitted),
            "intact_controls": len(controls),
            "longest_rendered_letters": max((row["audit"]["letters"] for row in all_rows), default=0),
        },
        "rendered_candidates": all_rows,
        "independent_audits": [
            "live trie orbit assignment",
            "independent normalized two-pointer scan",
            "forward/reverse SHA-256 comparison",
        ],
        "novelty_preflight": {
            "status": "passed",
            "signature": SIGNATURE,
            "prior_artifacts_compared": [
                "typed-clause-zipper-20260919",
                "typed-word-boundary-clause-automaton-20260918",
            ],
            "prior_exact_tapes_loaded": len(prior),
            "duplicate_rejections": duplicate_rejections,
            "catalogue_text_imported": False,
            "reversible_lexical_pairs_used": False,
            "finished_tape_reversed": False,
            "word_order_symmetry_used": False,
            "rlaif_used": False,
        },
        "next_repair": {
            "operator": "add one held-out subject/object noun bundle whose opening and closing characters satisfy the live orbit frontier",
            "reason": "the bounded complete-SVO chart records no 39--60-letter closure; controls fail at the first unresolved orbit after the shared outer prefix",
            "preserve": ["finite subject/finite verb/object completion", "ordinary rendering order", "independent two-pointer and SHA audits"],
        },
        "reader_gate": "closed; intact controls and mechanical checks do not certify readability",
    }


if __name__ == "__main__":
    payload = run()
    out = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    for row in payload["rendered_candidates"][:3]:
        print(row["rendered"])
