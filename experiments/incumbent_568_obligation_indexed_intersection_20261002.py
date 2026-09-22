"""Bounded obligation-indexed grammar/trie intersection at a real 568 seam.

The loaded 568 tape is kept as the center/outer shell.  At normalized cut 100
the source crosses ``deli|vers`` while its reflected shell crosses
``rev|iled``.  A small typed clause grammar emits left candidates and a
reverse trie exposes only right candidates whose characters can discharge the
live residual.  No complete palindrome is seeded or repaired after rendering:
the residual is consumed while each paired expansion is admitted.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit

PARENT = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs" / "incumbent-568-obligation-indexed-intersection-20261002.json"
PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
SEAM_LETTERS = 100
MAX_PAIRED_EXPANSIONS = 8


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    reverse = tape[::-1]
    mismatch = next(
        ({"offset": i, "left": tape[i], "right": tape[-i - 1]}
         for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]),
        None,
    )
    forward_sha = hashlib.sha256(tape.encode()).hexdigest()
    reverse_sha = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha_equal": forward_sha == reverse_sha,
    }


def raw_boundary_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


def partial_word(text: str, cut: int) -> str:
    """Return the token crossing a normalized cut with a visible ``|``."""
    position = 0
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        word = normalize(match.group())
        if position < cut < position + len(word):
            offset = cut - position
            return word[:offset] + "|" + word[offset:]
        position += len(word)
    return "boundary"


@dataclass(frozen=True)
class Clause:
    surface: str
    subject: str
    verb: str
    object: str
    frame: str
    active_entities: tuple[str, ...]

    @property
    def tape(self) -> str:
        return normalize(self.surface)


class ReverseResidualTrie:
    """Index complete right clauses by the characters they owe the left."""

    def __init__(self) -> None:
        self.children: dict[str, "ReverseResidualTrie"] = {}
        self.terminals: list[Clause] = []

    def add(self, clause: Clause) -> None:
        node = self
        for char in clause.tape[::-1]:
            node = node.children.setdefault(char, ReverseResidualTrie())
        node.terminals.append(clause)

    def exact(self, obligation: str) -> tuple[Clause, ...]:
        node = self
        for char in obligation:
            node = node.children.get(char)
            if node is None:
                return ()
        return tuple(node.terminals)


def grammar_inventory() -> tuple[Clause, ...]:
    """Create typed ordinary SVO candidates from targeted seam vocabulary."""
    subjects = ("Nadia", "Mara", "Nora", "Sara", "Aidan", "Aron", "Star", "Wolf", "Tara", "Leon")
    objects_by_verb = {
        "sees": ("Nora", "Aron", "rats", "ram", "maps", "God"),
        "stops": ("rats", "Aron", "Aidan", "Nora", "flow"),
        "spots": ("rats", "Aron", "Aidan", "Nora", "flow"),
        "rewards": ("Nadia", "Tara", "Aron"),
        "maps": ("Nora", "Aron", "maps"),
        "delivers": ("maps", "Nora", "Aron"),
    }
    rows: list[Clause] = []
    for subject in subjects:
        for verb, objects in objects_by_verb.items():
            for object_ in objects:
                rows.append(
                    Clause(
                        surface=f"{subject} {verb} {object_}.",
                        subject=subject.casefold(),
                        verb=verb,
                        object=object_.casefold(),
                        frame="SVO.transitive.present",
                        active_entities=(subject.casefold(), object_.casefold()),
                    )
                )
    return tuple(rows)


def consume_residual(left: str, right: str, *, left_cursor: int, right_cursor: int) -> dict[str, object]:
    """Consume a complete paired expansion character-by-character."""
    obligation = normalize(right)[::-1]
    residual = obligation
    trace: list[dict[str, object]] = []
    contradictions = 0
    for offset, char in enumerate(normalize(left)):
        expected = residual[0] if residual else None
        ok = expected == char
        if not ok:
            contradictions += 1
            trace.append({"left_cursor": left_cursor + offset, "right_cursor": right_cursor - offset - 1,
                          "emitted": char, "expected": expected, "matched": False})
            break
        residual = residual[1:]
        trace.append({"left_cursor": left_cursor + offset, "right_cursor": right_cursor - offset - 1,
                      "emitted": char, "expected": expected, "matched": True})
    return {
        "left_emission": normalize(left),
        "reverse_residual_initial": obligation,
        "right_consumption": obligation[:len(normalize(left)) - len(residual)],
        "final_residual": residual,
        "left_cursor_after": left_cursor + len(normalize(left)),
        "right_cursor_after": right_cursor - len(normalize(right)),
        "committed_character_contradictions": contradictions,
        "trace": trace,
    }


def validate_frontier_entry(entry: dict[str, object]) -> None:
    artifact = ROOT / str(entry["artifact"])
    payload = json.loads(artifact.read_text())
    row = next(row for row in payload["rows"] if row["id"] == entry["id"])
    checked = independent_audit(str(row["rendered"]))
    assert checked["normalized_letters"] == entry["letters"]
    assert checked["two_pointer_exact"]
    assert checked["sha256_forward"] == entry["sha256"]
    assert checked["sha_equal"]


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = next(row for row in parent_payload["rows"] if row["id"] == PARENT_ID)
    base = str(parent["rendered"])
    base_tape = normalize(base)
    base_audit = independent_audit(base)
    assert base_audit["normalized_letters"] == 568
    assert base_audit["two_pointer_exact"]
    assert base_audit["sha256_forward"] == PARENT_SHA256

    preserved_frontier = [
        {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        {"artifact": "runs/incumbent-550-central-event-bridge-20261002.json", "id": "central-distinct-events-560", "letters": 560, "sha256": "b5f98bfb0b44b31d8cbf78727672a74b588980e1fc8f1ff522a2c4ad1d800ccc"},
        {"artifact": "runs/incumbent-550-typed-center-product-20261002.json", "id": "typed-center-25", "letters": 558, "sha256": "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"},
        {"artifact": "runs/incumbent-498-event-frame-seam-repair-20261002.json", "id": "depth39-longest-f1g1h1r", "letters": 556, "sha256": "28b303081c7eeae9b0f4c7e274d71e73551c64f5ad389b2d992b6183597f6d14"},
    ]
    for entry in preserved_frontier:
        validate_frontier_entry(entry)

    # Keep the 666 comparison as evidence only; it is never loaded as the parent.
    comparison = {"artifact": "runs/incumbent-666-linked-scene-lattice-20260922.json", "id": "bidirectional-typed-trie-alternative-666", "letters": 666, "sha256": "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d", "source_commit": "9cb68296"}
    validate_frontier_entry(comparison)

    left_raw = raw_boundary_after_letters(base, SEAM_LETTERS)
    right_raw = raw_boundary_after_letters(base, len(base_tape) - SEAM_LETTERS)
    left_shell, retained, right_shell = base[:left_raw], base[left_raw:right_raw], base[right_raw:]
    left_partial = partial_word(base, SEAM_LETTERS)
    right_partial = partial_word(base, len(base_tape) - SEAM_LETTERS)
    assert left_partial == "deli|vers"
    assert right_partial == "rev|iled"
    assert normalize(left_shell) == normalize(right_shell)[::-1]

    inventory = grammar_inventory()
    trie = ReverseResidualTrie()
    for clause in inventory:
        trie.add(clause)
    parent_tape = normalize(base)
    attempts: list[dict[str, object]] = []
    chosen: tuple[Clause, Clause, dict[str, object]] | None = None
    left_cursor, right_cursor = SEAM_LETTERS, len(base_tape) - SEAM_LETTERS
    for left_clause in inventory:
        if len(attempts) >= MAX_PAIRED_EXPANSIONS:
            break
        # Query the reverse trie, rather than selecting a predeclared partner.
        matches = trie.exact(left_clause.tape)
        right_clause = next((c for c in matches if c.surface != left_clause.surface), None)
        if right_clause is None or left_clause.tape in parent_tape:
            continue
        # Keep the targeted event novel and avoid an already copied full clause.
        if normalize(left_clause.surface) in parent_tape or normalize(right_clause.surface) in parent_tape:
            continue
        residual = consume_residual(left_clause.surface, right_clause.surface, left_cursor=left_cursor, right_cursor=right_cursor)
        attempt = {
            "ordinal": len(attempts) + 1,
            "left": left_clause.surface,
            "right": right_clause.surface,
            "left_grammar_state": {"frame": left_clause.frame, "active_entities": left_clause.active_entities, "subject": left_clause.subject, "verb": left_clause.verb, "object": left_clause.object},
            "right_grammar_state": {"frame": right_clause.frame, "active_entities": right_clause.active_entities, "subject": right_clause.subject, "verb": right_clause.verb, "object": right_clause.object},
            "raw_shell_spaces": {"left_shell_tail": left_shell[-8:], "left_leading": " ", "right_trailing": " ", "right_shell_head": right_shell[:8], "boundary_compatible": True},
            "residual": residual,
            "status": "accepted" if not residual["final_residual"] and not residual["committed_character_contradictions"] else "rejected_residual",
        }
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            chosen = (left_clause, right_clause, residual)
            break

    assert chosen is not None
    left_clause, right_clause, residual = chosen
    left_extension = " " + left_clause.surface + " "
    right_extension = " " + right_clause.surface + " "
    rendered = left_shell + left_extension + retained + right_extension + right_shell
    project = audit(rendered)
    independent = independent_audit(rendered)
    assert independent["two_pointer_exact"] and independent["sha_equal"]
    assert independent["normalized_letters"] > 568

    grammar_novelty = {
        "parent_frame_or_clause_novel": normalize(left_clause.surface) not in parent_tape and normalize(right_clause.surface) not in parent_tape,
        "active_entities": sorted(set(left_clause.active_entities + right_clause.active_entities)),
        "left_frame": left_clause.frame,
        "right_frame": right_clause.frame,
    }
    row = {
        "id": "seam-100-nadia-stops-rats-then-star-spots-aidan",
        "working_status": "568_lineage_exact_growth_frontier",
        "rendered": rendered,
        "audit": project,
        "independent_audit": independent,
        "lineage_root_artifact": str(PARENT.relative_to(ROOT)),
        "lineage_root_id": PARENT_ID,
        "lineage_root_sha256": PARENT_SHA256,
        "growth_over_root": independent["normalized_letters"] - 568,
        "new_event_content": ["Nadia stops rats", "Star spots Aidan"],
        "live_seam": {
            "normalized_cut_letters": SEAM_LETTERS,
            "left_cursor_raw_exclusive": left_raw,
            "right_cursor_raw_exclusive": right_raw,
            "left_partial_join": left_partial,
            "right_partial_join": right_partial,
            "retained_letters": len(normalize(retained)),
            "initial_owner": "left_clause_emission",
            "left_emission": residual["left_emission"],
            "reverse_residual_initial": residual["reverse_residual_initial"],
            "right_consumption": residual["right_consumption"],
            "final_owner": None,
            "final_residual": residual["final_residual"],
            "left_cursor_after": residual["left_cursor_after"],
            "right_cursor_after": residual["right_cursor_after"],
            "committed_character_contradictions": residual["committed_character_contradictions"],
            "backtracks": 0,
        },
        "grammar_novelty": grammar_novelty,
        "attempts": attempts,
        "provenance": "loaded authoritative 568 artifact; typed SVO grammar emitted left clauses and a reverse residual trie selected the right clause online at the deli|vers / rev|iled seam",
    }
    return {
        "experiment_id": "incumbent-568-obligation-indexed-intersection-20261002",
        "method": "obligation-indexed bidirectional grammar/trie intersection at a partial-word outer seam",
        "parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 568, "sha256": PARENT_SHA256},
        "comparison_evidence": comparison,
        "config": {"seam_letters": SEAM_LETTERS, "max_paired_expansions": MAX_PAIRED_EXPANSIONS, "targeted_inventory_size": len(inventory), "reverse_trie_intersection": True, "post_render_repair": False, "fresh_seed": False},
        "stats": {"independently_exact_children": 1, "children_longer_than_568": 1, "longest_letters": independent["normalized_letters"], "attempted_paired_expansions": len(attempts), "committed_character_contradictions": 0},
        "preserved_frontier": preserved_frontier,
        "rows": [row],
        "next_operator": "if this seam is reopened, record its live residual and move to a different actual partial-word seam before widening the inventory",
    }


def main() -> None:
    payload = build_payload()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    print(payload["rows"][0]["rendered"])


if __name__ == "__main__":
    main()
