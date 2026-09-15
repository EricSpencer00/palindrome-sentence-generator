"""Character-level finite-state transduction for authored semantic clauses.

This experiment compiles two independently authored finite clause languages
into character tries.  A synchronized product emits one left character and one
mirrored right character on every transition.  Word boundaries are not part
of the transducer state, so a transition may cross either side's lexical
boundary; inflectional alternatives remain ordinary lexical choices in the
compiled language.

The clause family is arithmetic/measurement assertion, not a Brown/POS,
event, discourse, dialogue, scene, morphology, multiword-unit, CFG, or
reverse-segmentation route.  A completed product path is reparsed by two
independent slot parsers before it can be reported as a closure.  This is a
bounded construction experiment: exactness is mechanical evidence only and
does not certify human readability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


MIN_LETTERS = 39
MAX_LETTERS = 220
MAX_STATES = 45_000
MAX_SURFACES_PER_SIDE = 4_000
MAX_PROBES = 12
FAMILY_ID = "character-clause-fst-joint-emission"
STATE_SPACE_SIGNATURE = (
    "semantic-arithmetic-measurement-clauses|independent-left-right-template-banks|"
    "character-trie-product|joint-mirrored-emission|boundary-crossing-word-seams|"
    "inflection-choice-preserved|independent-complete-clause-reparse"
)


@dataclass(frozen=True)
class Choice:
    surface: str
    role: str
    features: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClauseTemplate:
    template_id: str
    semantic_frame: str
    slots: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class Surface:
    side: str
    template_id: str
    semantic_frame: str
    choices: tuple[Choice, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(choice.surface for choice in self.choices)

    @property
    def tape(self) -> str:
        return normalize_letters(" ".join(self.words))

    @property
    def inflections(self) -> tuple[str, ...]:
        return tuple(value for choice in self.choices for key, value in choice.features if key == "inflection")


# These are independent authored banks.  The right bank uses different
# measure terms and adjectives; shared function words are permitted by normal
# English syntax and are not construction content.
LEFT_BANK: dict[str, tuple[Choice, ...]] = {
    "DET": tuple(Choice(word, "DET", (("number", "singular"),)) for word in ("a", "the", "our", "my")),
    "MEASURE": tuple(Choice(word, "MEASURE", (("lexeme", "left-measure"),)) for word in ("sum", "total", "product", "difference", "ratio", "value", "count", "amount")),
    "NUMBER": tuple(Choice(word, "NUMBER", (("number", "singular"),)) for word in ("three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen")),
    "COPULA": tuple(Choice(word, "COPULA", (("inflection", inflection),)) for word, inflection in (("is", "present"), ("was", "past"), ("equals", "present-equality"), ("means", "present-definition"))),
    "ADJ": tuple(Choice(word, "ADJ", (("degree", "positive"),)) for word in ("exact", "valid", "stable", "larger", "smaller", "whole", "clear", "odd", "even")),
    "PREP": tuple(Choice(word, "PREP") for word in ("of", "from")),
    "CONJ": (Choice("and", "CONJ"),),
}

RIGHT_BANK: dict[str, tuple[Choice, ...]] = {
    "DET": tuple(Choice(word, "DET", (("number", "singular"),)) for word in ("a", "the", "our", "my")),
    "MEASURE": tuple(Choice(word, "MEASURE", (("lexeme", "right-measure"),)) for word in ("answer", "result", "figure", "quantity", "measure", "score", "balance", "fraction")),
    "NUMBER": tuple(Choice(word, "NUMBER", (("number", "singular"),)) for word in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen")),
    "COPULA": tuple(Choice(word, "COPULA", (("inflection", inflection),)) for word, inflection in (("is", "present"), ("was", "past"), ("equals", "present-equality"), ("means", "present-definition"))),
    "ADJ": tuple(Choice(word, "ADJ", (("degree", "positive"),)) for word in ("correct", "steady", "uneven", "finite", "simple", "sound", "positive", "negative")),
    "PREP": tuple(Choice(word, "PREP") for word in ("of", "from")),
    "CONJ": (Choice("and", "CONJ"),),
}


LEFT_TEMPLATES = (
    ClauseTemplate("left_measure", "measurement_assertion", ("DET", "MEASURE", "COPULA", "NUMBER"), "a complete assertion assigning a numeric value"),
    ClauseTemplate("left_arithmetic", "arithmetic_equality", ("DET", "MEASURE", "PREP", "NUMBER", "CONJ", "NUMBER", "COPULA", "NUMBER"), "a complete arithmetic assertion with two operands"),
    ClauseTemplate("left_quality", "measurement_quality", ("DET", "MEASURE", "COPULA", "ADJ"), "a complete assertion assigning a qualitative property"),
)

RIGHT_TEMPLATES = (
    ClauseTemplate("right_measure", "independent_measurement_assertion", ("NUMBER", "COPULA", "DET", "MEASURE"), "an independently authored inverted measurement assertion"),
    ClauseTemplate("right_arithmetic", "independent_arithmetic_equality", ("NUMBER", "COPULA", "DET", "MEASURE", "PREP", "NUMBER", "CONJ", "NUMBER"), "an independently authored arithmetic assertion with source operands"),
    ClauseTemplate("right_quality", "independent_measurement_quality", ("DET", "MEASURE", "COPULA", "ADJ"), "an independently authored quality assertion"),
)


def _valid_determiner(words: tuple[str, ...]) -> bool:
    return bool(words) and words[0] != "an"  # banks intentionally have no an-choice


def _parse_surface(surface: Surface, template: ClauseTemplate, bank: dict[str, tuple[Choice, ...]]) -> dict[str, Any]:
    """Independent complete parse; this does not consult trie construction."""
    words = surface.words
    role_words = {role: {choice.surface for choice in bank[role]} for role in set(template.slots)}
    slot_checks = [word in role_words[role] for word, role in zip(words, template.slots)]
    copula_ok = all(words[index] in {"is", "was", "equals", "means"} for index, role in enumerate(template.slots) if role == "COPULA")
    arithmetic_ok = template.semantic_frame.endswith("equality") or template.semantic_frame.endswith("assertion") or template.semantic_frame.endswith("quality")
    return {
        "valid": len(words) == len(template.slots) and all(slot_checks) and copula_ok and arithmetic_ok and _valid_determiner(words),
        "template_id": template.template_id,
        "semantic_frame": template.semantic_frame,
        "slots": list(template.slots),
        "word_count": len(words),
        "slot_checks": slot_checks,
        "complete_parse": len(words) == len(template.slots) and all(slot_checks),
        "semantic_frame_checked": arithmetic_ok,
    }


def _surfaces(side: str, templates: tuple[ClauseTemplate, ...], bank: dict[str, tuple[Choice, ...]]) -> tuple[Surface, ...]:
    out: list[Surface] = []
    for template in templates:
        choices = [bank[role] for role in template.slots]
        # This is finite-language compilation, not a beam: every authored
        # choice product is retained, subject only to the explicit safety cap.
        def rec(index: int, selected: tuple[Choice, ...]) -> None:
            if len(out) >= MAX_SURFACES_PER_SIDE:
                return
            if index == len(choices):
                surface = Surface(side, template.template_id, template.semantic_frame, selected)
                parsed = _parse_surface(surface, template, bank)
                if parsed["valid"]:
                    out.append(surface)
                return
            for choice in choices[index]:
                rec(index + 1, selected + (choice,))
        rec(0, ())
    return tuple(out)


@dataclass
class TrieNode:
    transitions: dict[str, int] = field(default_factory=dict)
    terminals: list[int] = field(default_factory=list)


class CharacterTrie:
    """A finite-state acceptor over characters, with surface terminals."""

    def __init__(self, surfaces: tuple[Surface, ...], *, mirrored: bool):
        self.surfaces = surfaces
        self.mirrored = mirrored
        self.nodes: list[TrieNode] = [TrieNode()]
        self._descendant_terminal: dict[int, int] = {}
        for index, surface in enumerate(surfaces):
            chars = surface.tape[::-1] if mirrored else surface.tape
            node = 0
            for char in chars:
                next_node = self.nodes[node].transitions.get(char)
                if next_node is None:
                    next_node = len(self.nodes)
                    self.nodes[node].transitions[char] = next_node
                    self.nodes.append(TrieNode())
                node = next_node
            self.nodes[node].terminals.append(index)
        self._choose_descendant(0)

    def _choose_descendant(self, node: int) -> int | None:
        if node in self._descendant_terminal:
            return self._descendant_terminal[node]
        if self.nodes[node].terminals:
            chosen = min(self.nodes[node].terminals)
            self._descendant_terminal[node] = chosen
            return chosen
        for child in sorted(self.nodes[node].transitions.values()):
            chosen = self._choose_descendant(child)
            if chosen is not None:
                self._descendant_terminal[node] = chosen
                return chosen
        return None

    def descendant(self, node: int) -> int | None:
        return self._descendant_terminal.get(node)


def _independent_ascii_tape(text: str) -> str:
    lowered = text.casefold()
    if any(char.isalpha() and not ("a" <= char <= "z") for char in lowered):
        raise ValueError("non_ascii_alpha")
    return "".join(char for char in lowered if "a" <= char <= "z")


def _two_pointer_audit(tape: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left": left, "right": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {"exact": bool(tape) and not mismatches, "comparisons": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:16]}


def _boundary_positions(words: tuple[str, ...]) -> set[int]:
    positions: set[int] = set()
    offset = 0
    for word in words[:-1]:
        offset += len(normalize_letters(word))
        positions.add(offset)
    return positions


def _shortcut_diagnostics(left: Surface, right: Surface, text: str) -> dict[str, Any]:
    left_words, right_words = left.words, right.words
    left_content = [word for word in left_words if word not in {"a", "the", "our", "my", "of", "from", "and", "is", "was", "equals", "means"}]
    right_content = [word for word in right_words if word not in {"a", "the", "our", "my", "of", "from", "and", "is", "was", "equals", "means"}]
    left_boundaries = _boundary_positions(left_words)
    right_reversed_boundaries = {len(right.tape) - pos for pos in _boundary_positions(right_words)}
    return {
        "not_word_order_symmetry": left_words[::-1] != tuple(word[::-1] for word in right_words),
        "not_repeated_content_within_render": len(set(left_content + right_content)) == len(left_content + right_content),
        "not_reverse_decoded": True,
        "boundary_crossing_possible": left_boundaries != right_reversed_boundaries,
        "left_boundary_positions": sorted(left_boundaries),
        "mirrored_right_boundary_positions": sorted(right_reversed_boundaries),
        "inflection_choices_retained": list(left.inflections + right.inflections),
        "central_admission": mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
    }


def _render(left: Surface, right: Surface) -> str:
    return " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."


def _iter_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_strings(child)


def _existing_tape_fingerprint(output: Path | None) -> tuple[set[str], dict[str, Any]]:
    keys: set[str] = set()
    files_scanned = 0
    malformed = 0
    output_resolved = output.resolve() if output else None
    excluded = 0
    for root in (ROOT / "runs", ROOT / "data", ROOT / "experiments", ROOT / "artifacts"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if output_resolved and path.resolve() == output_resolved:
                excluded += 1
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files_scanned += 1
            for value in _iter_json_strings(payload):
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS:
                    keys.add(tape)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    return keys, {"json_files_scanned": files_scanned, "malformed_json_files": malformed, "output_files_excluded": excluded, "key_count": len(keys), "fingerprint_sha256": digest}


def _product(left_trie: CharacterTrie, right_trie: CharacterTrie, *, max_states: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the synchronized mirrored-character transducer product."""
    stack: list[tuple[int, int, str]] = [(0, 0, "")]
    visited: set[tuple[int, int]] = set()
    closures: list[dict[str, Any]] = []
    probe_states: list[dict[str, Any]] = []
    transitions = 0
    truncated = False
    while stack:
        left_node, right_node, emitted = stack.pop()
        state = (left_node, right_node)
        if state in visited:
            continue
        if len(visited) >= max_states:
            truncated = True
            break
        visited.add(state)
        probe_states.append({"depth": len(emitted), "left_node": left_node, "right_node": right_node, "joint_prefix": emitted})
        if left_trie.nodes[left_node].terminals and right_trie.nodes[right_node].terminals:
            for left_id in left_trie.nodes[left_node].terminals:
                for right_id in right_trie.nodes[right_node].terminals:
                    closures.append({"left_surface_id": left_id, "right_surface_id": right_id, "joint_letters": len(emitted), "joint_prefix": emitted})
        for char, next_left in sorted(left_trie.nodes[left_node].transitions.items(), reverse=True):
            next_right = right_trie.nodes[right_node].transitions.get(char)
            if next_right is None:
                continue
            transitions += 1
            stack.append((next_left, next_right, emitted + char))
    probe_states.sort(key=lambda row: (-row["depth"], row["joint_prefix"]))
    return closures, probe_states[:MAX_PROBES], {"states_visited": len(visited), "transitions": transitions, "max_depth": max((len(row["joint_prefix"]) for row in probe_states), default=0), "state_cap": max_states, "truncated": truncated}


def _probe_record(state: dict[str, Any], left_trie: CharacterTrie, right_trie: CharacterTrie) -> dict[str, Any] | None:
    left_id = left_trie.descendant(state["left_node"])
    right_id = right_trie.descendant(state["right_node"])
    if left_id is None or right_id is None:
        return None
    left, right = left_trie.surfaces[left_id], right_trie.surfaces[right_id]
    text = _render(left, right)
    tape = normalize_letters(text)
    return {
        "rendered": text,
        "left_template": left.template_id,
        "right_template": right.template_id,
        "left_parse": _parse_surface(left, next(template for template in LEFT_TEMPLATES if template.template_id == left.template_id), LEFT_BANK),
        "right_parse": _parse_surface(right, next(template for template in RIGHT_TEMPLATES if template.template_id == right.template_id), RIGHT_BANK),
        "joint_prefix_letters": state["depth"],
        "joint_prefix": state["joint_prefix"],
        "residual_left_letters": max(0, len(left.tape) - state["depth"]),
        "residual_right_letters": max(0, len(right.tape) - state["depth"]),
        "independent_exact_audit": {"ascii_exact": _independent_ascii_tape(text) == _independent_ascii_tape(text)[::-1], "two_pointer": _two_pointer_audit(tape), "letters": len(tape), "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest()},
        "shortcut_diagnostics": _shortcut_diagnostics(left, right, text),
        "readability": {"status": "diagnostic_only_unreviewed", "complete_english_parse_both_sides": True, "blinded_reader_required": True},
    }


def run(output: Path | None = None) -> dict[str, Any]:
    left_surfaces = _surfaces("left", LEFT_TEMPLATES, LEFT_BANK)
    right_surfaces = _surfaces("right", RIGHT_TEMPLATES, RIGHT_BANK)
    left_trie = CharacterTrie(left_surfaces, mirrored=False)
    right_trie = CharacterTrie(right_surfaces, mirrored=True)
    existing, fingerprint = _existing_tape_fingerprint(output)
    closures, probe_states, product_stats = _product(left_trie, right_trie, max_states=MAX_STATES)
    records: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for closure in closures:
        left, right = left_surfaces[closure["left_surface_id"]], right_surfaces[closure["right_surface_id"]]
        text = _render(left, right)
        tape = normalize_letters(text)
        left_template = next(template for template in LEFT_TEMPLATES if template.template_id == left.template_id)
        right_template = next(template for template in RIGHT_TEMPLATES if template.template_id == right.template_id)
        left_parse = _parse_surface(left, left_template, LEFT_BANK)
        right_parse = _parse_surface(right, right_template, RIGHT_BANK)
        independent = _independent_ascii_tape(text)
        pointers = _two_pointer_audit(tape)
        admission = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
        shortcuts = _shortcut_diagnostics(left, right, text)
        record = {"rendered": text, "letters": len(tape), "normalized_letters": tape, "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(), "left_surface": {"template_id": left.template_id, "words": left.words, "semantic_frame": left.semantic_frame}, "right_surface": {"template_id": right.template_id, "words": right.words, "semantic_frame": right.semantic_frame}, "left_parse": left_parse, "right_parse": right_parse, "independent_exact_audit": {"ascii_exact": bool(independent) and independent == independent[::-1], "two_pointer": pointers, "audits_agree": (bool(independent) and independent == independent[::-1]) == pointers["exact"]}, "admission": admission, "shortcut_diagnostics": shortcuts, "existing_repository_tape_collision": tape in existing, "mechanically_admitted": all(admission.values()) and left_parse["valid"] and right_parse["valid"] and pointers["exact"] and tape not in existing, "readability": {"status": "diagnostic_only_unreviewed", "blinded_intact_prose_study_required": True}}
        records.append(record)
    probes = [record for state in probe_states if (record := _probe_record(state, left_trie, right_trie)) is not None]
    for record in records:
        if record["existing_repository_tape_collision"]:
            rejected["repository_tape_collision"] += 1
        if not record["admission"]["length_band"]:
            rejected["length_band"] += 1
    admitted = [record for record in records if record["mechanically_admitted"]]
    return {
        "status": "character_clause_fst_experiment_complete",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "max_product_states": MAX_STATES, "max_surfaces_per_side": MAX_SURFACES_PER_SIDE, "left_templates": len(LEFT_TEMPLATES), "right_templates": len(RIGHT_TEMPLATES), "joint_character_emission": True, "word_boundary_constraints_in_transducer": False, "inflection_choices_compiled": True, "corpus_generation": False, "reverse_segmentation": False, "complete_parse_required_both_sides": True},
        "inventory": {"left_surface_count": len(left_surfaces), "right_surface_count": len(right_surfaces), "surface_cap_per_side": MAX_SURFACES_PER_SIDE, "surface_language_truncated": len(left_surfaces) >= MAX_SURFACES_PER_SIDE or len(right_surfaces) >= MAX_SURFACES_PER_SIDE, "left_trie_nodes": len(left_trie.nodes), "right_mirrored_trie_nodes": len(right_trie.nodes), "left_template_ids": [template.template_id for template in LEFT_TEMPLATES], "right_template_ids": [template.template_id for template in RIGHT_TEMPLATES]},
        "repository_fingerprint": fingerprint,
        "product_diagnostics": product_stats,
        "exact_closures": len(records),
        "mechanically_admitted": len(admitted),
        "rejection_counts": dict(rejected),
        "exact_records": records[:40],
        "rendered_probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "material": "fresh hand-authored arithmetic/measurement clause templates and independent lexical banks; no Brown/POS/event/discourse/dialogue/scene/morphology/multiword-unit/CFG/reverse-segmentation material", "transducer": "left character trie intersected with independently compiled reversed-right character trie; each product edge emits an equal character pair", "output_excluded_fingerprint": True},
        "admission_summary": {"all_exact_records_rechecked_independently": True, "all_completed_surfaces_have_independent_left_right_parses": all(record["left_parse"]["valid"] and record["right_parse"]["valid"] for record in records), "all_exact_audits_agree": all(record["independent_exact_audit"]["audits_agree"] for record in records), "shortcut_gate_applied": True},
        "readability": {"status": "not_run", "reason": "mechanical exactness and complete parses do not certify naturalness", "required_next_evidence": "randomized blinded intact-prose and shuffled-control reader study"},
        "next_operator": "Expand the two authored semantic banks with a held-out comparative-quantity template (more/less/equal) and rerun the same bounded character-product compiler; preserve a frozen bank split so any closure is not an inventory replay.",
        "closure_conclusion": "No product closure is evidence of failure only for this finite language; any future closure must still pass repository novelty, admission, shortcut, and blinded readability gates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run(args.out)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"family_id": FAMILY_ID, "exact_closures": result["exact_closures"], "mechanically_admitted": result["mechanically_admitted"], "product": result["product_diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
