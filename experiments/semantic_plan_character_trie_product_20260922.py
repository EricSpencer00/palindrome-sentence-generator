"""Semantic-plan-first product of two complete finite clause grammars.

This lane is deliberately independent of the productive-affix / ``r=s``
work.  A linked pair of event plans is committed before lexical emission.
Each plan is a complete finite grammar with agreement, valency, and an
optional typed adjunct.  The two realizers then walk per-slot character tries
from opposite ends.  They never build a bank of completed clauses.

The product carries the hard construction gates while it emits: content
lemmas are globally fresh, opposing internal word boundaries cannot coincide,
and any already-complete proper palindromic word span is rejected.  Exact
survivors are reparsed and audited by independent code before they are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)


EXPERIMENT_ID = "semantic-plan-character-trie-product-20260922"
DEFAULT_OUTPUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
WORD = re.compile(r"^[a-z]+$")


@dataclass(frozen=True)
class Lexeme:
    surface: str
    lemma: str

    def __post_init__(self) -> None:
        if not WORD.fullmatch(self.surface) or not WORD.fullmatch(self.lemma):
            raise ValueError((self.surface, self.lemma))


@dataclass(frozen=True)
class Slot:
    role: str
    choices: tuple[Lexeme, ...]
    function: bool = False


@dataclass(frozen=True)
class ClausePlan:
    plan_id: str
    side: str
    event: str
    discourse_relation: str
    parse_signature: str
    subject_number: str
    finite_number: str
    valency: str
    adjunct_type: str | None
    slots: tuple[Slot, ...]

    def grammar_record(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "side": self.side,
            "event": self.event,
            "discourse_relation": self.discourse_relation,
            "parse_signature": self.parse_signature,
            "subject_number": self.subject_number,
            "finite_number": self.finite_number,
            "agreement": self.subject_number == self.finite_number,
            "valency": self.valency,
            "adjunct_type": self.adjunct_type,
            "roles": [slot.role for slot in self.slots],
            "lexical_choices_by_role": {
                slot.role: len(slot.choices) for slot in self.slots
            },
        }


def _lex(words: str) -> tuple[Lexeme, ...]:
    return tuple(Lexeme(word, word) for word in words.split())


def _verbs(pairs: str) -> tuple[Lexeme, ...]:
    """Read ``surface:lemma`` pairs without deriving morphology in search."""
    rows = []
    for pair in pairs.split():
        surface, lemma = pair.split(":", 1)
        rows.append(Lexeme(surface, lemma))
    return tuple(rows)


# These are role lexicons, not phrase or palindrome inventories.  Every word
# is usable in ordinary non-palindromic realizations of its declared event.
OBSERVERS = _lex(
    "analysts auditors clerks crews guards inspectors monitors observers "
    "readers researchers scouts surveyors trackers workers"
)
OBSERVE = _verbs(
    "check:check compare:compare examine:examine inspect:inspect log:log "
    "mark:mark measure:measure monitor:monitor note:note record:record "
    "scan:scan study:study test:test track:track watch:watch"
)
EVIDENCE = _lex(
    "alerts charts changes clues entries faults gauges labels levels maps "
    "marks notes patterns readings records routes samples signals tracks"
)

HANDLERS = _lex(
    "artisans bakers builders carriers cleaners cooks crews farmers handlers "
    "keepers makers packers porters sorters stewards workers"
)
HANDLE = _verbs(
    "carry:carry clean:clean collect:collect count:count deliver:deliver "
    "file:file gather:gather keep:keep move:move pack:pack place:place "
    "prepare:prepare seal:seal sort:sort stack:stack store:store"
)
GOODS = _lex(
    "baskets boxes bread cards cartons crates files grain letters parcels "
    "parts samples seeds sheets stores tools trays"
)

REVIEWERS = _lex(
    "advisers analysts boards clerks councils editors panels readers reviewers "
    "teams testers trustees workers"
)
ASSESS = _verbs(
    "assess:assess confirm:confirm consider:consider debate:debate evaluate:evaluate "
    "judge:judge rate:rate review:review score:score test:test verify:verify"
)
PROPOSALS = _lex(
    "amendments budgets changes claims designs drafts estimates findings "
    "options plans policies proposals reports results schedules"
)

RESPONDERS = _lex(
    "assistants clerks crews guards keepers operators readers stewards teams "
    "workers"
)
RESPOND = _verbs(
    "answer:answer gather:gather pause:pause react:react reply:reply respond:respond "
    "return:return signal:signal speak:speak wait:wait work:work"
)
PLACES = _lex(
    "archives benches depots desks docks gates halls harbors kitchens markets "
    "offices sheds shelves stations stores tables yards"
)

STATES = _lex(
    "active alert calm careful complete clear early empty open orderly ready "
    "secure stable steady thorough"
)

FUNCTION = {
    "the": Lexeme("the", "the"),
    "a": Lexeme("a", "a"),
    "after": Lexeme("after", "after"),
    "because": Lexeme("because", "because"),
    "so": Lexeme("so", "so"),
    "then": Lexeme("then", "then"),
    "while": Lexeme("while", "while"),
    "at": Lexeme("at", "at"),
    "beside": Lexeme("beside", "beside"),
    "by": Lexeme("by", "by"),
    "near": Lexeme("near", "near"),
}


def _slot(role: str, choices: Iterable[Lexeme], *, function: bool = False) -> Slot:
    rows = tuple(choices)
    if not rows:
        raise ValueError(role)
    return Slot(role, rows, function)


def _left_plan(
    prefix: str,
    event: str,
    relation: str,
    subjects: tuple[Lexeme, ...],
    verbs: tuple[Lexeme, ...],
    objects: tuple[Lexeme, ...],
    *,
    adjunct: bool,
) -> ClausePlan:
    slots = [
        _slot("subject", subjects),
        _slot("finite_verb", verbs),
        _slot("object_determiner", (FUNCTION["the"],), function=True),
        _slot("required_object", objects),
    ]
    adjunct_type = None
    if adjunct:
        adjunct_type = "location"
        slots.extend((
            _slot("location_preposition", (FUNCTION["near"], FUNCTION["beside"], FUNCTION["at"]), function=True),
            _slot("location_determiner", (FUNCTION["the"],), function=True),
            _slot("location", PLACES),
        ))
    return ClausePlan(
        f"{prefix}-{'location' if adjunct else 'core'}", "left", event,
        relation, "plural-subject/transitive-finite/direct-object", "plural",
        "plural", "transitive-required-object", adjunct_type, tuple(slots),
    )


def _right_plan(
    prefix: str,
    event: str,
    relation: str,
    connector: str,
    *,
    copular: bool,
    adjunct: bool,
) -> ClausePlan:
    slots = [
        _slot("discourse_connector", (FUNCTION[connector],), function=True),
        _slot("subject", RESPONDERS),
    ]
    if copular:
        slots.extend((
            _slot("finite_copula", (Lexeme("are", "be"),), function=True),
            _slot("required_predicative_complement", STATES),
        ))
        signature = "connector/plural-subject/copular-finite/adjective-complement"
        valency = "copular-required-predicative-complement"
    else:
        slots.extend((
            _slot("finite_verb", RESPOND),
            _slot("required_location_preposition", (FUNCTION["at"], FUNCTION["by"], FUNCTION["near"]), function=True),
            _slot("location_determiner", (FUNCTION["the"],), function=True),
            _slot("required_location_complement", PLACES),
        ))
        signature = "connector/plural-subject/intransitive-finite/required-location"
        valency = "intransitive-required-location-complement"
    adjunct_type = None
    if adjunct:
        adjunct_type = "time"
        slots.extend((
            _slot("time_preposition", (FUNCTION["after"],), function=True),
            _slot("time_determiner", (FUNCTION["the"],), function=True),
            _slot("time_reference", _lex("audit check review shift test")),
        ))
    return ClausePlan(
        f"{prefix}-{'copular' if copular else 'locative'}-{'time' if adjunct else 'core'}",
        "right", event, relation, signature, "plural", "plural", valency,
        adjunct_type, tuple(slots),
    )


def semantic_plan_pairs() -> tuple[tuple[ClausePlan, ClausePlan], ...]:
    """Return linked event plans; this call precedes all lexical emission."""
    links = (
        ("observation-response", "observation", OBSERVERS, OBSERVE, EVIDENCE, "then", False),
        ("handling-readiness", "handling", HANDLERS, HANDLE, GOODS, "so", True),
        ("review-response", "review", REVIEWERS, ASSESS, PROPOSALS, "while", False),
    )
    rows = []
    for relation, event, subjects, verbs, objects, connector, copular in links:
        # Adjunct presence is a semantic-plan decision, never a lexical repair.
        for left_adjunct, right_adjunct in ((False, False), (True, False), (False, True), (True, True)):
            left = _left_plan(
                f"{event}-action", event, relation, subjects, verbs, objects,
                adjunct=left_adjunct,
            )
            right = _right_plan(
                f"{event}-consequence", f"{event}-consequence", relation,
                connector, copular=copular, adjunct=right_adjunct,
            )
            if left.event == right.event or left.parse_signature == right.parse_signature:
                raise AssertionError("plans must be semantically and syntactically distinct")
            rows.append((left, right))
    return tuple(rows)


class TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.terminal: Lexeme | None = None


def _compile_slot(slot: Slot, reverse: bool) -> TrieNode:
    root = TrieNode()
    for lexeme in slot.choices:
        node = root
        key = lexeme.surface[::-1] if reverse else lexeme.surface
        for character in key:
            node = node.children.setdefault(character, TrieNode())
        if node.terminal is not None:
            raise AssertionError((slot.role, lexeme.surface))
        node.terminal = lexeme
    return root


@dataclass(frozen=True)
class ArmState:
    slot_index: int
    prefix: str
    emitted_words: tuple[tuple[str, str, str, bool], ...]
    boundaries: tuple[int, ...]


class ClauseAutomaton:
    """Lazy per-slot trie automaton; it cannot enumerate whole clauses."""

    def __init__(self, plan: ClausePlan, *, reverse: bool) -> None:
        self.plan = plan
        self.reverse = reverse
        self.slots = plan.slots[::-1] if reverse else plan.slots
        self.roots = tuple(_compile_slot(slot, reverse) for slot in self.slots)

    @property
    def start(self) -> ArmState:
        return ArmState(0, "", (), ())

    def accepted(self, state: ArmState) -> bool:
        if state.slot_index != len(self.slots) - 1:
            return False
        node = self._node(state)
        return node.terminal is not None

    def _node(self, state: ArmState) -> TrieNode:
        node = self.roots[state.slot_index]
        for character in state.prefix:
            node = node.children[character]
        return node

    def _commit(self, state: ArmState, depth: int) -> ArmState | None:
        node = self._node(state)
        lexeme = node.terminal
        if lexeme is None:
            return None
        slot = self.slots[state.slot_index]
        word = (slot.role, lexeme.surface, lexeme.lemma, slot.function)
        return ArmState(
            state.slot_index + 1, "", state.emitted_words + (word,),
            state.boundaries + (depth,),
        )

    def emitting_edges(self, state: ArmState, depth: int) -> tuple[tuple[str, ArmState], ...]:
        """Return character-consuming edges after any word-boundary epsilon."""
        rows: list[tuple[str, ArmState]] = []
        node = self._node(state)
        for character in sorted(node.children):
            rows.append((character, ArmState(
                state.slot_index, state.prefix + character,
                state.emitted_words, state.boundaries,
            )))
        committed = self._commit(state, depth)
        if committed is not None and committed.slot_index < len(self.slots):
            rows.extend(self.emitting_edges(committed, depth))
        return tuple(rows)

    def finalize(self, state: ArmState, depth: int) -> ArmState | None:
        """Commit the last word after its last matched character."""
        if not self.accepted(state):
            return None
        node = self._node(state)
        lexeme = node.terminal
        assert lexeme is not None
        slot = self.slots[state.slot_index]
        word = (slot.role, lexeme.surface, lexeme.lemma, slot.function)
        return ArmState(
            len(self.slots), "", state.emitted_words + (word,),
            state.boundaries + (depth,),
        )


def _physical_words(state: ArmState, reverse: bool) -> tuple[tuple[str, str, str, bool], ...]:
    return state.emitted_words[::-1] if reverse else state.emitted_words


def _content_lemmas(state: ArmState) -> tuple[str, ...]:
    return tuple(
        lemma for _role, _surface, lemma, function in state.emitted_words
        if not function and lemma not in REPEATABLE_FUNCTION_WORDS
    )


def _has_palindromic_span(words: tuple[tuple[str, str, str, bool], ...], *, proper_only: bool) -> bool:
    surfaces = tuple(row[1] for row in words)
    for start in range(len(surfaces)):
        for stop in range(start + 2, len(surfaces) + 1):
            if proper_only and start == 0 and stop == len(surfaces):
                continue
            tape = "".join(surfaces[start:stop])
            if tape == tape[::-1]:
                return True
    return False


def online_gate(left: ArmState, right: ArmState, *, complete: bool) -> tuple[bool, str | None]:
    """Apply only irreversible gates to a product prefix."""
    lemmas = _content_lemmas(left) + _content_lemmas(right)
    if len(lemmas) != len(set(lemmas)):
        return False, "global_content_freshness"
    shared = set(left.boundaries).intersection(right.boundaries)
    allowed = {left.boundaries[-1]} if complete and left.boundaries and right.boundaries else set()
    if shared - allowed:
        return False, "complementary_boundary_mask"
    # The two known segments are separated by unselected words until closure.
    # A palindrome wholly inside either segment can never be repaired later.
    for arm, reverse in ((left, False), (right, True)):
        words = _physical_words(arm, reverse)
        if _has_palindromic_span(words, proper_only=False):
            return False, "proper_span_mask"
    return True, None


@dataclass(frozen=True)
class ProductState:
    left: ArmState
    right: ArmState
    depth: int
    matched: str


def _frontier_record(
    state: ProductState,
    left_machine: ClauseAutomaton,
    right_machine: ClauseAutomaton,
    *,
    failure: str,
    left_next: tuple[str, ...] = (),
    right_next: tuple[str, ...] = (),
) -> dict:
    left_slot = left_machine.slots[state.left.slot_index].role if state.left.slot_index < len(left_machine.slots) else "complete"
    right_slot = right_machine.slots[state.right.slot_index].role if state.right.slot_index < len(right_machine.slots) else "complete"
    return {
        "depth": state.depth,
        "matched_character_prefix": state.matched,
        "left": {
            "plan_id": left_machine.plan.plan_id,
            "slot": left_slot,
            "partial_token": state.left.prefix,
            "committed_words": [row[1] for row in _physical_words(state.left, False)],
            "boundary_cursors": list(state.left.boundaries),
            "next_characters": list(left_next),
        },
        "right_from_outer_edge": {
            "plan_id": right_machine.plan.plan_id,
            "slot": right_slot,
            "partial_reversed_token": state.right.prefix,
            "committed_physical_words": [row[1] for row in _physical_words(state.right, True)],
            "boundary_cursors": list(state.right.boundaries),
            "next_characters": list(right_next),
        },
        "failure": failure,
    }


def _render(left: ArmState, right: ArmState) -> str:
    left_words = [row[1] for row in _physical_words(left, False)]
    right_words = [row[1] for row in _physical_words(right, True)]
    return " ".join(left_words).capitalize() + "; " + " ".join(right_words) + "."


def _role_parse(plan: ClausePlan, words: tuple[str, ...]) -> dict:
    if len(words) != len(plan.slots):
        return {"ok": False, "reason": "slot_count"}
    chosen = []
    for slot, word in zip(plan.slots, words):
        lexeme = next((item for item in slot.choices if item.surface == word), None)
        if lexeme is None:
            return {"ok": False, "reason": f"role:{slot.role}"}
        chosen.append({"role": slot.role, "surface": word, "lemma": lexeme.lemma})
    required = {
        "subject",
        "finite_verb" if any(slot.role == "finite_verb" for slot in plan.slots) else "finite_copula",
    }
    roles = {slot.role for slot in plan.slots}
    valency_ok = (
        "required_object" in roles
        or "required_location_complement" in roles
        or "required_predicative_complement" in roles
    )
    return {
        "ok": required.issubset(roles) and valency_ok
        and plan.subject_number == plan.finite_number,
        "agreement": plan.subject_number == plan.finite_number,
        "valency_complete": valency_ok,
        "event": plan.event,
        "roles": chosen,
    }


def independent_validate(rendered: str, left_plan: ClausePlan, right_plan: ClausePlan) -> dict:
    """Re-tokenize, reparse, and replay exactness without product state."""
    units = tokenize(rendered)
    split = len(left_plan.slots)
    left_words = units[:split]
    right_words = units[split:]
    tape = "".join(re.findall(r"[a-z]", rendered.casefold()))
    mismatch = next(
        (index for index in range(len(tape) // 2) if tape[index] != tape[-1 - index]),
        None,
    )
    left_parse = _role_parse(left_plan, left_words)
    right_parse = _role_parse(right_plan, right_words)
    left_boundaries = []
    cursor = 0
    for word in left_words:
        cursor += len(word)
        left_boundaries.append(cursor)
    right_boundaries = []
    cursor = 0
    for word in reversed(right_words):
        cursor += len(word)
        right_boundaries.append(cursor)
    shared = sorted(set(left_boundaries).intersection(right_boundaries))
    terminal = [len("".join(left_words))] if len("".join(left_words)) == len("".join(right_words)) else []
    forbidden = [value for value in shared if value not in terminal]
    content = []
    for parsed in (left_parse, right_parse):
        for row in parsed.get("roles", ()):
            if row["lemma"] not in REPEATABLE_FUNCTION_WORDS:
                content.append(row["lemma"])
    proper_span = _has_palindromic_span(
        tuple(("", word, word, False) for word in units), proper_only=True,
    )
    forward_hash = hashlib.sha256(tape.encode()).hexdigest()
    reverse_hash = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward_hash,
        "sha256_reverse": reverse_hash,
        "hashes_agree": forward_hash == reverse_hash,
        "left_parse": left_parse,
        "right_parse": right_parse,
        "different_events": left_plan.event != right_plan.event,
        "different_parse_signatures": left_plan.parse_signature != right_plan.parse_signature,
        "global_content_freshness": len(content) == len(set(content)),
        "complementary_boundary_mask": {
            "left": left_boundaries,
            "right_from_outer_edge": right_boundaries,
            "shared": shared,
            "allowed_terminal": terminal,
            "forbidden_internal": forbidden,
            "passes": not forbidden,
        },
        "proper_span_mask": {"passes": not proper_span},
    }


def _search_pair(
    left_plan: ClausePlan,
    right_plan: ClausePlan,
    *,
    budget: int,
    max_survivors: int,
) -> dict:
    left_machine = ClauseAutomaton(left_plan, reverse=False)
    right_machine = ClauseAutomaton(right_plan, reverse=True)
    queue = deque((ProductState(left_machine.start, right_machine.start, 0, ""),))
    seen: set[tuple[ArmState, ArmState]] = set()
    stats: Counter = Counter()
    survivors = []
    deepest: list[dict] = []
    truncated = False

    def remember(record: dict) -> None:
        deepest.append(record)
        deepest.sort(key=lambda row: (-row["depth"], row["left"]["plan_id"], row["matched_character_prefix"]))
        del deepest[12:]

    while queue and stats["states_visited"] < budget and len(survivors) < max_survivors:
        state = queue.popleft()
        key = (state.left, state.right)
        if key in seen:
            continue
        seen.add(key)
        stats["states_visited"] += 1

        left_final = left_machine.finalize(state.left, state.depth)
        right_final = right_machine.finalize(state.right, state.depth)
        if left_final is not None and right_final is not None:
            ok, reason = online_gate(left_final, right_final, complete=True)
            if not ok:
                stats[f"online_reject_{reason}"] += 1
                remember(_frontier_record(
                    state, left_machine, right_machine,
                    failure=f"terminal_{reason}",
                ))
                continue
            rendered = _render(left_final, right_final)
            independent = independent_validate(rendered, left_plan, right_plan)
            central = mechanical_admission_checks(rendered, min_letters=39, max_letters=240)
            passed = (
                independent["letters"] > 38
                and independent["two_pointer_exact"]
                and independent["hashes_agree"]
                and independent["left_parse"]["ok"]
                and independent["right_parse"]["ok"]
                and independent["different_events"]
                and independent["different_parse_signatures"]
                and independent["global_content_freshness"]
                and independent["complementary_boundary_mask"]["passes"]
                and independent["proper_span_mask"]["passes"]
                and all(central.values())
            )
            row = {
                "rendered": rendered,
                "semantic_plan": {
                    "left": left_plan.grammar_record(),
                    "right": right_plan.grammar_record(),
                    "selected_before_lexical_emission": True,
                },
                "independent_validation": independent,
                "central_admission": central,
                "all_gates_pass": passed,
            }
            if passed:
                survivors.append(row)
                stats["independently_validated_survivors"] += 1
            else:
                stats["terminal_gate_rejections"] += 1
            continue

        left_edges = left_machine.emitting_edges(state.left, state.depth)
        right_edges = right_machine.emitting_edges(state.right, state.depth)
        right_by_char: dict[str, list[ArmState]] = {}
        for character, next_state in right_edges:
            right_by_char.setdefault(character, []).append(next_state)
        matched_edges = 0
        for character, next_left in left_edges:
            for next_right in right_by_char.get(character, ()):
                matched_edges += 1
                stats["character_matches"] += 1
                ok, reason = online_gate(next_left, next_right, complete=False)
                if not ok:
                    stats[f"online_reject_{reason}"] += 1
                    continue
                queue.append(ProductState(
                    next_left, next_right, state.depth + 1,
                    state.matched + character,
                ))
        stats["character_mismatches_pruned"] += len(left_edges) * len(right_edges) - matched_edges
        if not matched_edges:
            stats["dead_frontiers"] += 1
            remember(_frontier_record(
                state, left_machine, right_machine,
                failure="disjoint_next_character_sets",
                left_next=tuple(sorted({row[0] for row in left_edges})),
                right_next=tuple(sorted({row[0] for row in right_edges})),
            ))

    if queue:
        truncated = True
        stats["queued_at_cap"] = len(queue)
        for state in sorted(queue, key=lambda row: -row.depth)[:4]:
            remember(_frontier_record(
                state, left_machine, right_machine,
                failure="state_budget_cap",
            ))
    return {
        "left_plan": left_plan.grammar_record(),
        "right_plan": right_plan.grammar_record(),
        "stats": dict(stats),
        "state_budget": budget,
        "truncated": truncated,
        "deepest_frontier": deepest,
        "survivors": survivors,
    }


def run(*, max_states: int = 250_000, max_survivors: int = 20) -> dict:
    started = time.monotonic()
    pairs = semantic_plan_pairs()
    pair_budget = max(1, max_states // len(pairs))
    rows = []
    all_survivors = []
    aggregate: Counter = Counter()
    for left_plan, right_plan in pairs:
        row = _search_pair(
            left_plan, right_plan, budget=pair_budget,
            max_survivors=max_survivors - len(all_survivors),
        )
        rows.append(row)
        aggregate.update(row["stats"])
        all_survivors.extend(row["survivors"])
        if len(all_survivors) >= max_survivors:
            break
    frontier = [item for row in rows for item in row["deepest_frontier"]]
    frontier.sort(key=lambda item: (-item["depth"], item["left"]["plan_id"], item["matched_character_prefix"]))
    deepest = frontier[:20]
    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "can two distinct complete finite clause parses, selected as a linked semantic event plan before lexicalization, "
            "close an exact shortcut-clean character palindrome above 38 letters?"
        ),
        "acceptance_gate": {
            "exact_letters": True,
            "minimum_letters_exclusive": 38,
            "two_different_complete_clause_parses": True,
            "agreement_and_required_valency": True,
            "semantic_event_link": True,
            "global_content_freshness_online": True,
            "complementary_internal_boundaries_disjoint_online": True,
            "proper_palindromic_spans_masked_online": True,
            "central_mechanical_admission": True,
            "independent_validation_every_survivor": True,
        },
        "construction": {
            "representation": "semantic-plan-first asynchronous per-slot character-trie product",
            "semantic_plan_selected_before_lexical_emission": True,
            "complete_phrase_bank_built": False,
            "completed_tapes_reversed": False,
            "word_boundaries_may_stagger": True,
            "fixed_palindrome_seed": False,
            "morphology_stack_or_r_equals_s": False,
            "proper_names": False,
            "catalogue_material": False,
            "fragments": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
        },
        "semantic_plan_pairs": [
            {"left": left.grammar_record(), "right": right.grammar_record()}
            for left, right in pairs
        ],
        "pair_runs": rows,
        "stats": {
            **dict(aggregate),
            "semantic_plan_pairs": len(pairs),
            "state_budget_total": max_states,
            "pair_budget": pair_budget,
            "exact_independently_validated_survivors": len(all_survivors),
            "max_survivor_letters": max(
                (row["independent_validation"]["letters"] for row in all_survivors),
                default=0,
            ),
            "deepest_character_cursor": max((row["depth"] for row in frontier), default=0),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "survivors": all_survivors,
        "deepest_semantic_character_frontier": deepest,
        "next_representation": (
            None if all_survivors else {
                "name": "packed synchronous dependency-chart product with delayed lexical completion",
                "concrete_change": (
                    "replace linear slot paths with chart items (head, dependent-set, agreement, valency, event, left-cursor, right-cursor, residual); "
                    "allow subject, object, and adjunct dependencies to complete after the character cursor crosses a clause-internal constituent boundary"
                ),
                "lexicon_policy": "freeze the exact role lexicons and semantic links from this run",
                "why_not_vocabulary_widening": (
                    "the saved frontier identifies a representation obstruction at trie/slot order; the successor changes dependency scheduling while holding words fixed"
                ),
            }
        ),
        "verdict": (
            "exact survivors require blinded human reading"
            if all_survivors
            else "bounded product found no admissible exact closure; deepest semantic/character cursor frontier retained"
        ),
        "provenance": {
            "host": os.uname().nodename,
            "python": sys.version.split()[0],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_validator": "fresh tokenization, role parse, outside-in two-pointer scan, and forward/reverse SHA-256",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-states", type=int, default=250_000)
    parser.add_argument("--max-survivors", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(max_states=args.max_states, max_survivors=args.max_survivors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "experiment_id": payload["experiment_id"],
        "host": payload["provenance"]["host"],
        "stats": payload["stats"],
        "survivors": [row["rendered"] for row in payload["survivors"]],
        "verdict": payload["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
