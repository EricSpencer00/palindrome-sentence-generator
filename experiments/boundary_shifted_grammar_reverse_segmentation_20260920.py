"""Boundary-shifted grammar-aware reverse segmentation.

The left side is generated as an ordinary complete clause from a fresh
POS/valency grammar.  Each character is then offered immediately to an
independently authored right-clause parser whose cursor starts at the right
edge and moves inward.  The parser carries the right grammar's word and
phrase roles while crossing optional-adjunct and multiword boundaries.

This is intentionally an online construction: it never materializes a left
tape and reverses it to create the right clause.  The right clause remains in
ordinary English order in the rendered result; only its parser cursor reads
that clause from the end.  Reversal appears only in the independent exact
audits after rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUT = ROOT / "runs/boundary-shifted-grammar-reverse-segmentation-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "boundary-shifted-grammar-reverse-segmentation-20260920"
SIGNATURE = (
    "boundary-shifted-grammar-reverse-segmentation|fresh-pos-valency-grammar|"
    "incremental-right-reverse-parser|optional-adjunct-boundary|"
    "live-word-boundary-state|independent-pointer-sha"
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_audit(text: str) -> dict:
    """Independent outside-in exact check, separate from construction."""
    tape = normalize(text)
    i, j = 0, len(tape) - 1
    mismatch: Optional[dict] = None
    while i < j:
        if tape[i] != tape[j]:
            mismatch = {
                "left_index": i,
                "right_index": j,
                "left_char": tape[i],
                "right_char": tape[j],
            }
            break
        i += 1
        j -= 1
    return {
        "normalized_tape": tape,
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
    }


def hash_audit(text: str) -> dict:
    """Independent forward/reverse SHA-256 audit after rendering."""
    tape = normalize(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def independent_audit(text: str) -> dict:
    pointer = pointer_audit(text)
    hashed = hash_audit(text)
    return {
        **pointer,
        **hashed,
        "independent_exact": pointer["exact"] and hashed["sha_equal_under_reversal"],
    }


@dataclass(frozen=True)
class Word:
    surface: str
    pos: str
    role: str
    phrase_index: int


@dataclass(frozen=True)
class Clause:
    frame_id: str
    bank: str
    production: str
    valency: str
    number: str
    tense: str
    words: tuple[Word, ...]
    grammar_slots: tuple[str, ...]
    roles: tuple[str, ...]

    @property
    def surface(self) -> str:
        return " ".join(word.surface for word in self.words)

    @property
    def letters(self) -> int:
        return len(normalize(self.surface))


@dataclass
class ReverseParserState:
    """Right grammar parser with a live character and word-boundary cursor."""

    words: tuple[Word, ...]
    expected_slots: tuple[str, ...]
    word_index: int = field(init=False)
    offset: int = field(init=False)
    slot_index: int = field(init=False)
    consumed: int = 0
    boundary_shifts: int = 0
    boundary_events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.word_index = len(self.words) - 1
        self.offset = len(self.words[-1].surface) - 1 if self.words else -1
        self.slot_index = len(self.expected_slots) - 1

    @property
    def done(self) -> bool:
        return self.word_index < 0

    def peek(self) -> Optional[str]:
        if self.done:
            return None
        return self.words[self.word_index].surface[self.offset]

    def location(self) -> Optional[dict]:
        if self.done:
            return None
        word = self.words[self.word_index]
        return {
            "word": word.surface,
            "pos": word.pos,
            "role": word.role,
            "word_index": self.word_index,
            "offset": self.offset,
            "word_length": len(word.surface),
            "expected_slot": self.expected_slots[self.slot_index]
            if 0 <= self.slot_index < len(self.expected_slots)
            else None,
        }

    def consume(self, expected_character: str) -> tuple[bool, Optional[dict]]:
        """Consume one incoming left character at the live right boundary."""
        if self.done:
            return False, {"reason": "right_parser_done"}
        word = self.words[self.word_index]
        expected_slot = self.expected_slots[self.slot_index]
        if word.role != expected_slot:
            return False, {
                "reason": "grammar_slot_mismatch",
                "word": word.surface,
                "word_role": word.role,
                "expected_slot": expected_slot,
            }
        actual = word.surface[self.offset]
        if actual != expected_character:
            return False, {
                "reason": "character_mismatch",
                "expected_character": expected_character,
                "right_character": actual,
                "right_location": self.location(),
            }

        event = None
        self.consumed += 1
        if self.offset == 0:
            prior_word = word
            prior_slot = expected_slot
            self.word_index -= 1
            self.slot_index -= 1
            if not self.done:
                self.boundary_shifts += 1
                self.offset = len(self.words[self.word_index].surface) - 1
                event = {
                    "from_word": prior_word.surface,
                    "from_role": prior_word.role,
                    "from_slot": prior_slot,
                    "to_word": self.words[self.word_index].surface,
                    "to_role": self.words[self.word_index].role,
                    "to_slot": self.expected_slots[self.slot_index],
                    "consumed_characters": self.consumed,
                }
            else:
                event = {
                    "from_word": prior_word.surface,
                    "from_role": prior_word.role,
                    "from_slot": prior_slot,
                    "to_word": None,
                    "to_role": None,
                    "to_slot": None,
                    "consumed_characters": self.consumed,
                }
            self.boundary_events.append(event)
        else:
            self.offset -= 1
        return True, event


@dataclass(frozen=True)
class Walk:
    status: str
    matched_characters: int
    online_equations: int
    first_mismatch: Optional[dict]
    right_boundary_shifts: int
    right_boundary_events: tuple[dict, ...]
    right_parser_location: Optional[dict]
    left_remaining: int
    right_remaining: int


def phrase_words(text: str, pos: str, role: str, phrase_index: int) -> tuple[Word, ...]:
    return tuple(Word(token, pos, role, phrase_index) for token in text.split())


# Fresh lexical banks: content words are disjoint between the left and right
# sides.  Function words may repeat because the clauses remain ordinary prose.
LEFT_SUBJECTS = {
    "sg": (("the amber historian", "historian"), ("a brisk violinist", "violinist")),
    "pl": (("amber historians", "historians"), ("brisk violinists", "violinists")),
}
LEFT_VERBS = {
    "sg": (("records", "record"), ("carries", "carry")),
    "pl": (("record", "record"), ("carry", "carry")),
}
LEFT_OBJECTS = {
    "sg": (("a weathered atlas", "atlas"), ("the bronze lamp", "lamp")),
    "pl": (("weathered atlases", "atlases"), ("bronze lamps", "lamps")),
}
LEFT_ADJUNCTS = (("near the river", "river"), ("under the arch", "arch"))

RIGHT_SUBJECTS = {
    "sg": (("the distant teacher", "teacher"), ("a watchful pilot", "pilot")),
    "pl": (("distant teachers", "teachers"), ("watchful pilots", "pilots")),
}
RIGHT_VERBS = {
    "sg": (("explains", "explain"), ("guards", "guard")),
    "pl": (("explain", "explain"), ("guard", "guard")),
}
RIGHT_OBJECTS = {
    "sg": (("a woven banner", "banner"), ("the hidden garden", "garden")),
    "pl": (("woven banners", "banners"), ("hidden gardens", "gardens")),
}
RIGHT_ADJUNCTS = (("through the meadow", "meadow"), ("beyond the harbor", "harbor"))


def build_clauses(bank: str) -> tuple[Clause, ...]:
    """Compile a small independent POS/valency grammar into complete clauses."""
    if bank == "left":
        subjects, verbs, objects, adjuncts = (
            LEFT_SUBJECTS, LEFT_VERBS, LEFT_OBJECTS, LEFT_ADJUNCTS
        )
    elif bank == "right":
        subjects, verbs, objects, adjuncts = (
            RIGHT_SUBJECTS, RIGHT_VERBS, RIGHT_OBJECTS, RIGHT_ADJUNCTS
        )
    else:
        raise ValueError(f"unknown grammar bank: {bank}")

    clauses: list[Clause] = []
    for number, subject_choices in subjects.items():
        for subject_text, subject_lemma in subject_choices:
            for verb_text, verb_lemma in verbs[number]:
                for object_text, object_lemma in objects[number]:
                    base_phrases = (
                        phrase_words(subject_text, "NP", "subject", 0),
                        phrase_words(verb_text, "V", "finite_verb", 1),
                        phrase_words(object_text, "NP", "theme", 2),
                    )
                    base_words = tuple(word for phrase in base_phrases for word in phrase)
                    base_id = f"{bank}:transitive:{number}:{subject_lemma}:{verb_lemma}:{object_lemma}"
                    clauses.append(
                        Clause(
                            frame_id=f"{base_id}:bare",
                            bank=bank,
                            production="transitive_bare",
                            valency="transitive",
                            number=number,
                            tense="present",
                            words=base_words,
                            grammar_slots=("subject", "finite_verb", "theme"),
                            roles=("subject", "finite_verb", "theme"),
                        )
                    )
                    for adjunct_text, adjunct_lemma in adjuncts:
                        adjunct_phrase = phrase_words(adjunct_text, "PP", "locative_adjunct", 3)
                        words = base_words + adjunct_phrase
                        clauses.append(
                            Clause(
                                frame_id=f"{base_id}:locative:{adjunct_lemma}",
                                bank=bank,
                                production="transitive_locative",
                                valency="transitive_plus_locative",
                                number=number,
                                tense="present",
                                words=words,
                                grammar_slots=(
                                    "subject", "finite_verb", "theme", "locative_adjunct"
                                ),
                                roles=(
                                    "subject", "finite_verb", "theme", "locative_adjunct"
                                ),
                            )
                        )
    return tuple(clauses)


def incremental_reverse_parse(left: Clause, right: Clause) -> Walk:
    """Feed left characters directly into the right parser's reverse cursor."""
    parser = ReverseParserState(right.words, right.roles)
    left_word_index = 0
    left_offset = 0
    matched = 0
    equations = 0
    first_mismatch = None

    while left_word_index < len(left.words) and not parser.done:
        left_word = left.words[left_word_index]
        left_character = left_word.surface[left_offset]
        right_character = parser.peek()
        equations += 1
        if right_character is None:
            first_mismatch = {
                "reason": "right_parser_done",
                "left_character": left_character,
                "left_location": {
                    "word": left_word.surface,
                    "role": left_word.role,
                    "word_index": left_word_index,
                    "offset": left_offset,
                },
            }
            break
        ok, detail = parser.consume(left_character)
        if not ok:
            first_mismatch = {
                "left_character": left_character,
                "left_location": {
                    "word": left_word.surface,
                    "role": left_word.role,
                    "word_index": left_word_index,
                    "offset": left_offset,
                },
                "right_location": parser.location(),
                **(detail or {}),
            }
            break
        matched += 1
        if left_offset == len(left_word.surface) - 1:
            left_word_index += 1
            left_offset = 0
        else:
            left_offset += 1

    left_remaining = 0
    if left_word_index < len(left.words):
        left_remaining = len(left.words[left_word_index].surface) - left_offset
        left_remaining += sum(len(word.surface) for word in left.words[left_word_index + 1 :])
    right_remaining = sum(len(word.surface) for word in parser.words[: parser.word_index + 1])
    if not parser.done and parser.word_index >= 0:
        right_remaining += parser.offset + 1

    if left_remaining == 0 and right_remaining == 0 and first_mismatch is None:
        status = "closed"
    elif first_mismatch is not None:
        status = "mismatch_pruned"
    else:
        status = "overhang_pruned"
    return Walk(
        status=status,
        matched_characters=matched,
        online_equations=equations,
        first_mismatch=first_mismatch,
        right_boundary_shifts=parser.boundary_shifts,
        right_boundary_events=tuple(parser.boundary_events),
        right_parser_location=parser.location(),
        left_remaining=left_remaining,
        right_remaining=right_remaining,
    )


def render_pair(left: Clause, right: Clause) -> str:
    left_text = left.surface[:1].upper() + left.surface[1:]
    return f"{left_text}; {right.surface}."


STOPWORDS = {
    "a", "an", "the", "near", "under", "through", "beyond", "and", "at", "by",
    "in", "on", "of",
}


def shortcut_flags(left: Clause, right: Clause) -> dict:
    def content(clause: Clause) -> set[str]:
        return {
            word.surface.casefold()
            for word in clause.words
            if word.surface.casefold() not in STOPWORDS
        }

    repeated = sorted(content(left).intersection(content(right)))
    self_palindromic = sorted(
        {
            word.surface.casefold()
            for word in (*left.words, *right.words)
            if len(normalize(word.surface)) > 1
            and normalize(word.surface) == normalize(word.surface)[::-1]
        }
    )
    return {
        "identical_clause_surface": left.surface == right.surface,
        "repeated_content_units": bool(repeated),
        "repeated_content_words": repeated,
        "self_palindromic_units": bool(self_palindromic),
        "self_palindromic_words": self_palindromic,
        "word_order_symmetry": left.roles == right.roles and left.frame_id == right.frame_id,
        "catalogue_text": False,
        "fragment": False,
        "finished_tape_reversal": False,
        "post_hoc_repair": False,
        "per_search_rlaif": False,
    }


def row_for(left: Clause, right: Clause, walk: Walk) -> dict:
    rendered = render_pair(left, right)
    audit = independent_audit(rendered)
    return {
        "rendered": rendered,
        "letters": audit["letters"],
        "audit": audit,
        "reverse_parser": {
            "status": walk.status,
            "matched_characters": walk.matched_characters,
            "online_equations": walk.online_equations,
            "first_mismatch": walk.first_mismatch,
            "right_boundary_shifts": walk.right_boundary_shifts,
            "right_boundary_events": list(walk.right_boundary_events),
            "right_parser_location": walk.right_parser_location,
            "left_remaining": walk.left_remaining,
            "right_remaining": walk.right_remaining,
        },
        "provenance": {
            "left_frame_id": left.frame_id,
            "right_frame_id": right.frame_id,
            "left_bank": left.bank,
            "right_bank": right.bank,
            "left_production": left.production,
            "right_production": right.production,
            "left_valency": left.valency,
            "right_valency": right.valency,
            "left_number": left.number,
            "right_number": right.number,
            "left_grammar_slots": list(left.grammar_slots),
            "right_grammar_slots": list(right.grammar_slots),
            "left_roles": list(left.roles),
            "right_roles": list(right.roles),
            "left_clause_words": [word.surface for word in left.words],
            "right_clause_words": [word.surface for word in right.words],
            "fresh_left_pos_valency_grammar": True,
            "fresh_right_pos_valency_grammar": True,
            "content_lexical_banks_disjoint": True,
            "right_boundaries_selected_during_parse": True,
            "left_characters_fed_incrementally": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
            "self_palindromic_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "fragment_output": False,
            "per_search_rlaif": False,
        },
        "shortcut_flags": shortcut_flags(left, right),
        "reader_facing_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "human_raters": 0,
            "reason": "No exact-clean closure from this bounded lane.",
        },
    }


def select_controls(observations: list[tuple], limit: int = 20) -> list[dict]:
    """Select complete prose across both optional-adjunct productions."""
    ordered = sorted(
        observations,
        key=lambda item: (
            shortcut_flags(item[2], item[3])["repeated_content_units"],
            -(item[2].letters + item[3].letters),
            item[2].production,
            item[3].production,
            item[0],
            item[1],
        ),
    )
    selected = []
    seen_surfaces: set[tuple[str, str]] = set()
    seen_productions: set[tuple[str, str]] = set()
    for left_index, right_index, left, right, walk in ordered:
        key = (left.surface, right.surface)
        productions = (left.production, right.production)
        if left.surface == right.surface or key in seen_surfaces:
            continue
        if productions in seen_productions and len(selected) < 4:
            continue
        selected.append((left_index, right_index, left, right, walk))
        seen_surfaces.add(key)
        seen_productions.add(productions)
        if len(selected) >= limit:
            break
    return [row_for(left, right, walk) for _, _, left, right, walk in selected]


def boundary_transition_contract() -> dict:
    """Exercise a genuine right-word boundary without making a candidate.

    The probe uses role-tagged alphabetic tokens only to test the parser state:
    the normal right surface is ``d cba`` and its inward stream is ``abcd``.
    It is deliberately stored as a cursor contract, never as reader prose.
    """
    left = Clause(
        frame_id="boundary-probe-left",
        bank="probe",
        production="probe",
        valency="probe",
        number="sg",
        tense="present",
        words=(
            Word("abc", "X", "first", 0),
            Word("d", "X", "second", 1),
        ),
        grammar_slots=("first", "second"),
        roles=("first", "second"),
    )
    right = Clause(
        frame_id="boundary-probe-right",
        bank="probe",
        production="probe",
        valency="probe",
        number="sg",
        tense="present",
        words=(
            Word("d", "X", "first", 0),
            Word("cba", "X", "second", 1),
        ),
        grammar_slots=("first", "second"),
        roles=("first", "second"),
    )
    walk = incremental_reverse_parse(left, right)
    return {
        "description": "Parser-only contract; not a sentence candidate or reader control.",
        "left_words": [word.surface for word in left.words],
        "right_words_in_normal_order": [word.surface for word in right.words],
        "status": walk.status,
        "matched_characters": walk.matched_characters,
        "right_boundary_shifts": walk.right_boundary_shifts,
        "boundary_events": list(walk.right_boundary_events),
        "expected": {
            "status": "closed",
            "right_boundary_shifts": 1,
            "normal_right_surface": "d cba",
            "inward_parse_stream": "abc d",
        },
    }


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    collisions = [
        entry.get("id")
        for entry in entries
        if entry.get("id") == EXPERIMENT_ID or entry.get("signature") == SIGNATURE
    ]
    related = [
        entry.get("id")
        for entry in entries
        if any(marker in entry.get("signature", "") for marker in (
            "reverse-segmentation", "boundary-shift", "reverse-automaton",
        ))
    ]
    return {
        "status": "passed" if not collisions else "collision",
        "registry_inspected": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": collisions,
        "related_signatures_seen": related,
        "signature": SIGNATURE,
        "distinction": (
            "Unlike prior fixed-tape segmentation and lexical reverse-trie lanes, "
            "this run feeds each character from a freshly generated ordinary left "
            "clause directly into an independently authored right POS/valency grammar. "
            "The right parser carries live word/phrase boundaries and an optional "
            "locative slot; no finished tape is reversed and no semordnilap unit is paired."
        ),
        "not_a_duplicate_sweep": True,
    }


def run() -> dict:
    left_clauses = build_clauses("left")
    right_clauses = build_clauses("right")
    observations = []
    mismatch_prunes = overhang_prunes = closures = 0
    equations = boundary_shifts = 0
    exact_candidates = []
    for left_index, left in enumerate(left_clauses):
        for right_index, right in enumerate(right_clauses):
            walk = incremental_reverse_parse(left, right)
            observations.append((left_index, right_index, left, right, walk))
            equations += walk.online_equations
            boundary_shifts += walk.right_boundary_shifts
            mismatch_prunes += walk.status == "mismatch_pruned"
            overhang_prunes += walk.status == "overhang_pruned"
            closures += walk.status == "closed"
            if walk.status == "closed":
                row = row_for(left, right, walk)
                if row["audit"]["independent_exact"]:
                    exact_candidates.append(row)

    exact_clean = [
        row for row in exact_candidates
        if row["letters"] > 38
        and not row["shortcut_flags"]["identical_clause_surface"]
        and not row["shortcut_flags"]["repeated_content_units"]
        and not row["shortcut_flags"]["self_palindromic_units"]
        and not row["shortcut_flags"]["word_order_symmetry"]
        and not row["shortcut_flags"]["fragment"]
    ]
    controls = select_controls(observations)
    boundary_contract = boundary_transition_contract()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Boundary-shifted grammar-aware reverse segmentation: generate a complete "
            "ordinary left POS/valency clause, feed its characters online to an "
            "independent right-clause grammar parser whose cursor consumes the right "
            "surface inward, and retain live optional-adjunct word boundaries."
        ),
        "operator_added": {
            "name": "incremental right reverse parser with optional locative boundary",
            "left_grammar": "NP(subject[number]) V(transitive[number]) NP(theme[number]) [PP(locative)]",
            "right_grammar": "NP(subject[number]) V(transitive[number]) NP(theme[number]) [PP(locative)]",
            "right_parser_input": "one left character at a time; no materialized reversed tape",
            "boundary_state": ["word_index", "character_offset", "grammar_slot_index", "boundary_shifts"],
            "fresh_pos_valency_grammars": True,
            "content_banks_disjoint": True,
            "semordnilap_word_pairing": False,
        },
        "stats": {
            "left_clause_paths": len(left_clauses),
            "right_clause_paths": len(right_clauses),
            "paired_grammar_states": len(observations),
            "online_character_equations": equations,
            "mismatch_prunes": mismatch_prunes,
            "overhang_prunes": overhang_prunes,
            "seam_closures": closures,
            "right_boundary_shifts": boundary_shifts,
            "boundary_contract_shifts": boundary_contract["right_boundary_shifts"],
            "rendered_controls": len(controls),
            "mechanical_exact_candidates": len(exact_candidates),
            "exact_clean_above_38": len(exact_clean),
            "longest_rendered_control_letters": max((row["letters"] for row in controls), default=0),
            "longest_exact_clean_letters": max((row["letters"] for row in exact_clean), default=0),
        },
        "rendered_controls": controls,
        "exact_candidates": exact_candidates,
        "exact_clean_candidates": exact_clean,
        "reader_facing_candidates": [],
        "reader_gate": {
            "status": "closed",
            "reason": "No exact-clean candidate from this lane has blinded human readability evidence.",
            "human_raters": 0,
        },
        "boundary_transition_contract": boundary_contract,
        "novelty_preflight": novelty_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "left_grammar_source": "fresh authored POS/valency slots",
            "right_grammar_source": "fresh independently authored POS/valency slots",
            "independent_audits": ["literal outside-in pointer scan", "forward/reverse SHA-256"],
            "right_reverse_cursor_is_construction_state": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "semordnilap_word_pairs": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "fragment_output": False,
            "word_order_symmetry": False,
            "per_search_rlaif": False,
        },
        "status": "completed_no_exact_closure" if not exact_candidates else "mechanical_exact_requires_reader_gate",
        "next_construction": {
            "operator": "typed right ditransitive boundary state",
            "description": (
                "Keep the online reverse parser and fresh left grammar fixed, then add "
                "one held-out right-side recipient/theme ditransitive production with an "
                "explicit valency slot transition. This is a new grammar state, not a "
                "finished-tape reversal, lexical repair, or semordnilap sweep."
            ),
            "reader_facing_test": (
                "Retain only complete prose; independently audit any exact closure above "
                "38 letters, then randomize intact versus shuffled controls for blinded ratings."
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"experiment_id": output["experiment_id"], "stats": output["stats"]}, sort_keys=True))
