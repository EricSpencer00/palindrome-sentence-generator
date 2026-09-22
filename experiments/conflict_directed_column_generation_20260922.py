"""Conflict-directed column generation over complete semantic clauses.

Two finite scene plans are fixed before solving.  Each plan starts with one
column per constituent.  CP-SAT selects columns and enforces every mirrored
character equality.  On UNSAT, an assumption core is reduced to a
cardinality-minimum conflict, converted to a constituent nogood, and sent to
a dynamic contextual lexical oracle.  The oracle builds fixed character tries
once from common Brown words that WordNet places in the existing slot's
syntactic and semantic category.  It is queried only at slot-local offsets
named by a core.  Every exposed column is reparsed in its complete clause
before the next solve.

This is deliberately a bounded obstruction experiment, not a sentence-bank
sweep: there are exactly two plans, at most eight solves per plan, and at most
two reparsed columns per conflict response.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import platform
from pathlib import Path
import re
import socket
from typing import Any

try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:  # Artifact-only inspection does not need the solver.
    cp_model = None

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "conflict-directed-column-generation-20260922"
PREFLIGHT_SIGNATURE = (
    "conflict-directed-column-generation|unsat-core-mirror-positions|"
    "contextual-constituent-oracle|packed-sentence-lattice|"
    "palindrome-equality-cp-sat|complete-clause-only|reader-gates"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / EXPERIMENT_ID / "search.json"
MAX_ITERATIONS = 8
MAX_ADDITIONS_PER_CORE = 2
MIN_BROWN_COUNT = 2
MAX_ORACLE_ENTRIES_PER_SLOT = 512
MAX_QUERY_TERMINALS = 32


@dataclass(frozen=True)
class Column:
    id: str
    slot: str
    text: str
    category: str
    number: str = "na"
    valency: str = "na"
    semantic_type: str = "na"
    lemma: str = ""
    initial: bool = False

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


@dataclass(frozen=True)
class Clause:
    id: str
    slots: tuple[str, ...]
    subject: str
    verb: str
    object: str
    prefix: str


@dataclass(frozen=True)
class ScenePlan:
    id: str
    description: str
    slots: tuple[str, ...]
    clauses: tuple[Clause, ...]
    pool: dict[str, tuple[Column, ...]]
    location_entity: str


@dataclass(frozen=True)
class OracleEntry:
    column: Column
    lexical_lemma: str
    brown_count: int
    brown_tags: tuple[str, ...]
    wordnet_synsets: tuple[str, ...]


class CharacterTrie:
    """A deterministic character trie with constrained arbitrary-offset lookup."""

    def __init__(self, entries: tuple[OracleEntry, ...]):
        self.root: dict[str, Any] = {}
        self.entries = entries
        for index, entry in enumerate(entries):
            node = self.root
            for char in entry.column.tape:
                node = node.setdefault(char, {})
            node.setdefault("$", []).append(index)

    def query(self, *, length: int, local_offset: int,
              required_chars: tuple[str, ...]) -> dict[str, Any]:
        if not 0 <= local_offset < length:
            return {"entries": [], "terminals_matching": 0, "nodes_visited": 0,
                    "truncated": False}
        required = frozenset(required_chars)
        hits: list[OracleEntry] = []
        nodes_visited = 0

        def walk(node: dict[str, Any], depth: int) -> None:
            nonlocal nodes_visited
            nodes_visited += 1
            if depth == length:
                hits.extend(self.entries[index] for index in node.get("$", ()))
                return
            keys = sorted(key for key in node if key != "$")
            if depth == local_offset:
                keys = [key for key in keys if key in required]
            for key in keys:
                walk(node[key], depth + 1)

        walk(self.root, 0)
        hits.sort(key=lambda item: (-item.brown_count, item.column.tape,
                                    item.column.id))
        return {"entries": hits[:MAX_QUERY_TERMINALS],
                "terminals_matching": len(hits), "nodes_visited": nodes_visited,
                "truncated": len(hits) > MAX_QUERY_TERMINALS}


@dataclass(frozen=True)
class LexicalOracle:
    tries: dict[tuple[str, str], CharacterTrie]
    entry_evidence: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def _col(plan: str, slot: str, text: str, category: str, *, initial: bool = False,
         number: str = "na", valency: str = "na", semantic_type: str = "na",
         lemma: str = "") -> Column:
    stem = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return Column(f"{plan}:{slot}:{stem}", slot, text, category, number,
                  valency, semantic_type, lemma, initial)


def build_plans() -> tuple[ScenePlan, ScenePlan]:
    """Return the same two fixed plans and their one-column controls."""
    plans = []

    plan = "orbital-meal"
    specs = {
        "opening": (
            _col(plan, "opening", "In orbit", "locative", initial=True, semantic_type="place", lemma="orbit"),
        ),
        "observer": (
            _col(plan, "observer", "the pilot", "np", initial=True, number="sg", semantic_type="human", lemma="pilot"),
        ),
        "observe": (
            _col(plan, "observe", "charts", "finite_verb", initial=True, number="sg", valency="transitive", lemma="chart"),
        ),
        "sky_object": (
            _col(plan, "sky_object", "the comet", "np", initial=True, number="sg", semantic_type="celestial", lemma="comet"),
        ),
        "anaphoric_link": (
            _col(plan, "anaphoric_link", "Later there", "discourse_anaphor", initial=True, semantic_type="place", lemma=""),
        ),
        "server": (
            _col(plan, "server", "the cook", "np", initial=True, number="sg", semantic_type="human", lemma="cook"),
        ),
        "serve": (
            _col(plan, "serve", "serves", "finite_verb", initial=True, number="sg", valency="transitive", lemma="serve"),
        ),
        "meal": (
            _col(plan, "meal", "zucchini", "np", initial=True, number="sg", semantic_type="food", lemma="zucchini"),
        ),
    }
    plans.append(ScenePlan(
        plan,
        "A pilot observes a comet from orbit; later a cook serves a meal at the same place.",
        tuple(specs),
        (Clause("observation", ("opening", "observer", "observe", "sky_object"),
                "observer", "observe", "sky_object", "opening"),
         Clause("meal_service", ("anaphoric_link", "server", "serve", "meal"),
                "server", "serve", "meal", "anaphoric_link")),
        specs, "orbit"))

    plan = "camp-supper"
    specs = {
        "opening": (
            _col(plan, "opening", "At sunset", "locative", initial=True, semantic_type="time_place", lemma="sunset"),
        ),
        "camper": (
            _col(plan, "camper", "the ranger", "np", initial=True, number="sg", semantic_type="human", lemma="ranger"),
        ),
        "light": (
            _col(plan, "light", "lights", "finite_verb", initial=True, number="sg", valency="transitive", lemma="light"),
        ),
        "camp_object": (
            _col(plan, "camp_object", "the camp", "np", initial=True, number="sg", semantic_type="camp", lemma="camp"),
        ),
        "anaphoric_link": (
            _col(plan, "anaphoric_link", "Later there", "discourse_anaphor", initial=True, semantic_type="place", lemma=""),
        ),
        "server": (
            _col(plan, "server", "the cook", "np", initial=True, number="sg", semantic_type="human", lemma="cook"),
        ),
        "serve": (
            _col(plan, "serve", "serves", "finite_verb", initial=True, number="sg", valency="transitive", lemma="serve"),
        ),
        "meal": (
            _col(plan, "meal", "salad", "np", initial=True, number="sg", semantic_type="food", lemma="salad"),
        ),
    }
    plans.append(ScenePlan(
        plan,
        "A ranger lights a camp at sunset; later a cook serves supper at that camp.",
        tuple(specs),
        (Clause("camp_setup", ("opening", "camper", "light", "camp_object"),
                "camper", "light", "camp_object", "opening"),
         Clause("supper", ("anaphoric_link", "server", "serve", "meal"),
                "server", "serve", "meal", "anaphoric_link")),
        specs, "camp"))
    return tuple(plans)  # type: ignore[return-value]


WORDNET_NOUN_ROOTS = {
    "human": ("person.n.01",),
    "place": ("location.n.01",),
    "time_place": ("time_period.n.01",),
    "celestial": ("celestial_body.n.01",),
    "food": ("food.n.01",),
    "camp": ("camp.n.01",),
}


def _brown_tag(raw_tag: str) -> str:
    """Discard Brown title/foreign suffixes without converting proper names."""
    return re.split(r"[-+]", raw_tag, maxsplit=1)[0]


def _descendant_synsets(wordnet, names: tuple[str, ...]) -> set[Any]:
    synsets: set[Any] = set()
    for name in names:
        root = wordnet.synset(name)
        synsets.add(root)
        synsets.update(root.closure(lambda item: item.hyponyms()))
    return synsets


def _verb_neighborhood(wordnet, lemma: str) -> set[Any]:
    """Use only the seed verb's WordNet senses and their direct siblings."""
    synsets = set(wordnet.synsets(lemma, pos=wordnet.VERB))
    for synset in tuple(synsets):
        for hypernym in synset.hypernyms():
            synsets.add(hypernym)
            synsets.update(hypernym.hyponyms())
    return synsets


def _surface_for_word(control: Column, word: str) -> str:
    lowered = control.text.casefold()
    if lowered.startswith("the "):
        return f"the {word}"
    if control.category == "locative":
        return f"{control.text.split()[0]} {word}"
    return word


def build_lexical_oracle(plans: tuple[ScenePlan, ...]) -> LexicalOracle:
    """Freeze Brown∩WordNet category tries before the first CP-SAT solve."""
    try:
        import nltk
        from nltk.corpus import brown, wordnet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The dynamic lexical oracle requires NLTK plus the Brown and WordNet corpora."
        ) from exc

    noun_counts: Counter[str] = Counter()
    verb_surface_counts: Counter[str] = Counter()
    tags_by_word: dict[str, set[str]] = {}
    brown_rows = 0
    for raw_word, raw_tag in brown.tagged_words():
        word = str(raw_word).casefold()
        if not word.isascii() or not word.isalpha():
            continue
        tag = _brown_tag(str(raw_tag))
        brown_rows += 1
        if tag == "NN":  # Plurals and proper-name tags are intentionally excluded.
            noun_counts[word] += 1
            tags_by_word.setdefault(word, set()).add(tag)
        if tag == "VBZ":
            verb_surface_counts[word] += 1
            tags_by_word.setdefault(word, set()).add(tag)

    tries: dict[tuple[str, str], CharacterTrie] = {}
    evidence: dict[str, dict[str, Any]] = {}
    digest_rows = []
    inventory_rows = []
    for plan in plans:
        for slot in plan.slots:
            control = plan.pool[slot][0]
            candidates: list[OracleEntry] = []
            if control.category == "finite_verb":
                eligible_synsets = _verb_neighborhood(wordnet, control.lemma)
                eligible_lemmas = set()
                lemma_synsets: dict[str, set[str]] = {}
                for synset in eligible_synsets:
                    for lexical_lemma in synset.lemmas():
                        name = lexical_lemma.name().casefold()
                        if not name.isascii() or not name.isalpha():
                            continue
                        frames = lexical_lemma.frame_strings()
                        if not any("something" in frame.casefold() for frame in frames):
                            continue
                        eligible_lemmas.add(name)
                        lemma_synsets.setdefault(name, set()).add(synset.name())
                for surface, count in verb_surface_counts.items():
                    lemma = wordnet.morphy(surface, wordnet.VERB)
                    if count < MIN_BROWN_COUNT or lemma not in eligible_lemmas:
                        continue
                    column = _col(plan.id, slot, surface, control.category,
                                  number=control.number, valency=control.valency,
                                  semantic_type=control.semantic_type, lemma=lemma)
                    candidates.append(OracleEntry(
                        column, lemma, count, tuple(sorted(tags_by_word[surface])),
                        tuple(sorted(lemma_synsets[lemma]))))
            elif control.category in {"np", "locative"} and control.semantic_type in WORDNET_NOUN_ROOTS:
                eligible_synsets = _descendant_synsets(
                    wordnet, WORDNET_NOUN_ROOTS[control.semantic_type])
                lemma_synsets: dict[str, set[str]] = {}
                for synset in eligible_synsets:
                    for lexical_lemma in synset.lemmas():
                        name = lexical_lemma.name().casefold()
                        if name.isascii() and name.isalpha():
                            lemma_synsets.setdefault(name, set()).add(synset.name())
                for word, count in noun_counts.items():
                    if count < MIN_BROWN_COUNT or word not in lemma_synsets:
                        continue
                    surface = _surface_for_word(control, word)
                    column = _col(plan.id, slot, surface, control.category,
                                  number=control.number, valency=control.valency,
                                  semantic_type=control.semantic_type, lemma=word)
                    candidates.append(OracleEntry(
                        column, word, count, tuple(sorted(tags_by_word[word])),
                        tuple(sorted(lemma_synsets[word]))))

            # A fixed, frequency-ranked cap makes the trie inventory reproducible;
            # no later conflict may widen it.
            candidates.sort(key=lambda item: (-item.brown_count, item.column.tape,
                                              item.column.id))
            unique: list[OracleEntry] = []
            seen_tapes = set()
            for item in candidates:
                if item.column.tape in seen_tapes:
                    continue
                seen_tapes.add(item.column.tape)
                unique.append(item)
                if len(unique) >= MAX_ORACLE_ENTRIES_PER_SLOT:
                    break
            entries = tuple(unique)
            tries[(plan.id, slot)] = CharacterTrie(entries)
            for item in entries:
                row = {
                    "column_id": item.column.id, "plan": plan.id, "slot": slot,
                    "surface": item.column.text, "tape": item.column.tape,
                    "category": item.column.category,
                    "semantic_type": item.column.semantic_type,
                    "number": item.column.number, "valency": item.column.valency,
                    "lexical_lemma": item.lexical_lemma,
                    "brown_count": item.brown_count,
                    "brown_tags": list(item.brown_tags),
                    "wordnet_synsets": list(item.wordnet_synsets),
                }
                evidence[item.column.id] = row
                digest_rows.append(row)
            inventory_rows.append({"plan": plan.id, "slot": slot,
                                   "entries": len(entries),
                                   "trie_tapes": len(seen_tapes)})

    inventory_blob = json.dumps(digest_rows, sort_keys=True, separators=(",", ":"))
    brown_blob = "\n".join(
        f"{word}\t{noun_counts[word]}\t{verb_surface_counts[word]}"
        for word in sorted(set(noun_counts) | set(verb_surface_counts))
    )
    provenance = {
        "construction": "fixed pre-solve character tries by existing plan and slot",
        "lexical_intersection": "Brown common lowercase lexical items intersected with WordNet category membership",
        "minimum_brown_count": MIN_BROWN_COUNT,
        "proper_name_tags_allowed": False,
        "max_entries_per_slot": MAX_ORACLE_ENTRIES_PER_SLOT,
        "max_query_terminals": MAX_QUERY_TERMINALS,
        "variable_length_policy": (
            "disabled: a query uses the active slot length, so every CP-SAT offset "
            "and mirror equality remains valid"
        ),
        "nltk_version": nltk.__version__,
        "wordnet_version": wordnet.get_version(),
        "brown_tagged_ascii_rows": brown_rows,
        "brown_index_sha256": hashlib.sha256(brown_blob.encode()).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_blob.encode()).hexdigest(),
        "inventory": inventory_rows,
    }
    return LexicalOracle(tries, evidence, provenance)


def render(plan: ScenePlan, selected: dict[str, Column]) -> str:
    surfaces = []
    for clause in plan.clauses:
        words = [selected[slot].text for slot in clause.slots]
        surfaces.append(" ".join(words) + ".")
    text = " ".join(surfaces)
    return text[0].upper() + text[1:]


def parse_complete_plan(plan: ScenePlan, selected: dict[str, Column]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    clause_rows = []
    for clause in plan.clauses:
        subject, verb, obj, prefix = (selected[clause.subject], selected[clause.verb],
                                      selected[clause.object], selected[clause.prefix])
        local = {
            "all_slots_present": all(slot in selected for slot in clause.slots),
            "finite_clause": verb.category == "finite_verb",
            "agreement": subject.number == verb.number,
            "valency": verb.valency == "transitive" and obj.category == "np",
            "attachment": prefix.category in {"locative", "discourse_anaphor"},
        }
        checks.update({f"{clause.id}:{key}": value for key, value in local.items()})
        clause_rows.append({"clause_id": clause.id, "surface": " ".join(
            selected[slot].text for slot in clause.slots) + ".", "checks": local})
    checks["anaphora"] = selected[plan.clauses[1].prefix].category == "discourse_anaphor"
    checks["shared_scene"] = bool(plan.location_entity and checks["anaphora"])
    lemmas = [selected[slot].lemma for slot in plan.slots if selected[slot].lemma]
    checks["fresh_content_lemmas"] = len(lemmas) == len(set(lemmas))
    checks["complete_clause_only"] = len(clause_rows) == 2 and all(
        row["checks"]["all_slots_present"] and row["checks"]["finite_clause"]
        for row in clause_rows)
    return {"accepted": all(checks.values()), "checks": checks, "clauses": clause_rows,
            "scene_continuity": {"location_entity": plan.location_entity,
                                 "anaphor": selected[plan.clauses[1].prefix].text}}


def independent_audit(text: str) -> dict[str, Any]:
    tape = normalize_letters(text)
    mismatch = next(({"position": i, "left": tape[i], "right": tape[-1-i]}
                     for i in range(len(tape) // 2) if tape[i] != tape[-1-i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized": tape, "letters": len(tape),
            "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse}


def packed_dag(plan: ScenePlan, active: dict[str, list[Column]]) -> dict[str, Any]:
    nodes = [{"slot": slot, "column_ids": [c.id for c in active[slot]],
              "tape_lengths": sorted({len(c.tape) for c in active[slot]})}
             for slot in plan.slots]
    edges = [{"from": plan.slots[i], "to": plan.slots[i + 1]}
             for i in range(len(plan.slots) - 1)]
    return {"start": plan.slots[0], "finish": plan.slots[-1], "nodes": nodes,
            "edges": edges, "represented_paths": _product(len(active[s]) for s in plan.slots)}


def _product(values):
    out = 1
    for value in values:
        out *= value
    return out


def char_layout(plan: ScenePlan, active: dict[str, list[Column]]) -> list[dict[str, Any]]:
    layout = []
    cursor = 0
    for slot in plan.slots:
        lengths = {len(column.tape) for column in active[slot]}
        if len(lengths) != 1:
            raise AssertionError(
                f"variable-length active columns require rebuilding the model layout: {slot} {lengths}"
            )
        for local in range(next(iter(lengths))):
            layout.append({"position": cursor, "slot": slot, "local_offset": local})
            cursor += 1
    return layout


def _build_model(plan: ScenePlan, active: dict[str, list[Column]], nogoods: list[dict[str, Any]],
                 equalities: tuple[int, ...] | None, assumptions: bool = False):
    if cp_model is None:
        raise RuntimeError(
            "OR-Tools is required to run the column generator; install the pinned requirements."
        )
    model = cp_model.CpModel()
    selects = {slot: model.new_int_var(0, len(active[slot]) - 1, f"select_{slot}")
               for slot in plan.slots}
    # Agreement and valency are represented as exact allowed feature tuples.
    for clause in plan.clauses:
        s, v, o = selects[clause.subject], selects[clause.verb], selects[clause.object]
        allowed_sv = [(i, j) for i, a in enumerate(active[clause.subject])
                      for j, b in enumerate(active[clause.verb]) if a.number == b.number]
        allowed_vo = [(i, j) for i, a in enumerate(active[clause.verb])
                      for j, b in enumerate(active[clause.object])
                      if a.valency == "transitive" and b.category == "np"]
        model.add_allowed_assignments([s, v], allowed_sv)
        model.add_allowed_assignments([v, o], allowed_vo)
    for nogood in nogoods:
        slots = nogood["slots"]
        old = [tuple(range(nogood["old_domain_sizes"][slot])) for slot in slots]
        model.add_forbidden_assignments([selects[slot] for slot in slots], itertools.product(*old))

    layout = char_layout(plan, active)
    chars = []
    for pos in layout:
        values = [ord(column.tape[pos["local_offset"]]) for column in active[pos["slot"]]]
        char = model.new_int_var(ord("a"), ord("z"), f"char_{pos['position']}")
        model.add_element(selects[pos["slot"]], values, char)
        chars.append(char)
    positions = tuple(range(len(chars) // 2)) if equalities is None else equalities
    assumption_map = {}
    for position in positions:
        if assumptions:
            literal = model.new_bool_var(f"mirror_{position}")
            model.add(chars[position] == chars[-1 - position]).only_enforce_if(literal)
            model.add_assumption(literal)
            assumption_map[literal.index] = position
        else:
            model.add(chars[position] == chars[-1 - position])
    return model, selects, layout, assumption_map


def _solve(model, selects=None):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.solve(model)
    picked = ({slot: solver.value(var) for slot, var in selects.items()}
              if selects and status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None)
    return solver, status, picked


def minimum_core(plan: ScenePlan, active: dict[str, list[Column]], nogoods: list[dict[str, Any]]) -> dict[str, Any]:
    assumed, _, layout, mapping = _build_model(plan, active, nogoods, None, assumptions=True)
    solver, status, _ = _solve(assumed)
    if status != cp_model.INFEASIBLE:
        raise AssertionError("core requested for feasible model")
    sufficient = tuple(sorted(mapping[index] for index in solver.sufficient_assumptions_for_infeasibility()))
    universe = tuple(range(len(layout) // 2))
    minimum = None
    checks = 0
    # Search sufficient-core positions first, then the full universe if needed.
    for domain in (sufficient, universe):
        for width in range(1, len(domain) + 1):
            for subset in itertools.combinations(domain, width):
                model, _, _, _ = _build_model(plan, active, nogoods, subset)
                _, substatus, _ = _solve(model)
                checks += 1
                if substatus == cp_model.INFEASIBLE:
                    minimum = subset
                    break
            if minimum is not None:
                break
        if minimum is not None:
            break
    assert minimum is not None
    positions = []
    implicated = set()
    for position in minimum:
        left, right = layout[position], layout[-1 - position]
        implicated.update((left["slot"], right["slot"]))
        positions.append({"position": position, "mirror_position": len(layout) - 1 - position,
                          "left": left, "right": right})
    # Proper subsets of a minimum-cardinality conflict are necessarily feasible;
    # record the explicit deletion checks for independent replay.
    proper = []
    for index in range(len(minimum)):
        subset = minimum[:index] + minimum[index + 1:]
        model, _, _, _ = _build_model(plan, active, nogoods, subset)
        _, substatus, _ = _solve(model)
        proper.append({"positions": list(subset), "feasible": substatus != cp_model.INFEASIBLE})
    return {"sufficient_assumption_positions": list(sufficient),
            "minimum_cardinality": len(minimum), "minimum_positions": positions,
            "implicated_slots": sorted(implicated), "subset_solves": checks,
            "proper_subset_checks": proper}


def contextual_columns(plan: ScenePlan, active: dict[str, list[Column]], core: dict[str, Any],
                       anchor: dict[str, Column], oracle: LexicalOracle
                       ) -> tuple[list[Column], dict[str, Any]]:
    """Query only the slot/offset obligations present in this minimum core."""
    queries = []
    for core_index, position in enumerate(core["minimum_positions"]):
        for side, opposite in (("left", "right"), ("right", "left")):
            target = position[side]
            mirror = position[opposite]
            required_chars = tuple(sorted({
                column.tape[mirror["local_offset"]]
                for column in active[mirror["slot"]]
            }))
            lengths = {len(column.tape) for column in active[target["slot"]]}
            if len(lengths) != 1:
                raise AssertionError("oracle query requires a rebuilt consistent slot layout")
            required_length = next(iter(lengths))
            query_id = f"{plan.id}:core-query:{core_index}:{side}"
            trie = oracle.tries[(plan.id, target["slot"])]
            result = trie.query(length=required_length,
                                local_offset=target["local_offset"],
                                required_chars=required_chars)
            queries.append({
                "query_id": query_id,
                "core_position": position["position"],
                "direction": side,
                "slot": target["slot"],
                "local_offset": target["local_offset"],
                "required_chars_from_active_mirror_domain": list(required_chars),
                "mirror_slot": mirror["slot"],
                "mirror_local_offset": mirror["local_offset"],
                "required_tape_length": required_length,
                "syntactic_category": anchor[target["slot"]].category,
                "semantic_category": anchor[target["slot"]].semantic_type,
                "number": anchor[target["slot"]].number,
                "valency": anchor[target["slot"]].valency,
                "terminals_matching": result["terminals_matching"],
                "nodes_visited": result["nodes_visited"],
                "truncated": result["truncated"],
                "returned_column_ids": [item.column.id for item in result["entries"]],
                "_entries": result["entries"],
            })

    added: list[Column] = []
    reparses = []
    rejections = []
    evaluated = set()
    active_ids = {column.id for columns in active.values() for column in columns}
    maximum_width = max((len(query["_entries"]) for query in queries), default=0)
    # Round-robin makes both sides of a core eligible before lower-ranked hits
    # from either side; the global two-column cap remains unchanged.
    for rank in range(maximum_width):
        for query in queries:
            if rank >= len(query["_entries"]):
                continue
            entry = query["_entries"][rank]
            column = entry.column
            base = {"query_id": query["query_id"], "rank": rank,
                    "column_id": column.id, "slot": column.slot,
                    "surface": column.text, "tape": column.tape,
                    "source": oracle.entry_evidence[column.id]}
            if column.id in active_ids:
                rejections.append(base | {"reason": "already_active"})
                continue
            if column.id in evaluated:
                rejections.append(base | {"reason": "duplicate_across_core_queries"})
                continue
            evaluated.add(column.id)
            if len(added) >= MAX_ADDITIONS_PER_CORE:
                rejections.append(base | {"reason": "two_column_core_cap_reached"})
                continue
            trial = dict(anchor)
            trial[column.slot] = column
            parse = parse_complete_plan(plan, trial)
            clause_spec = next(item for item in plan.clauses if column.slot in item.slots)
            clause = next(row for row in parse["clauses"]
                          if row["clause_id"] == clause_spec.id)
            row = base | {
                "full_clause_surface": clause["surface"],
                "full_clause_checks": clause["checks"],
                "full_plan_accepted": parse["accepted"],
                "checks": parse["checks"],
                "decision": "add" if parse["accepted"] else "reject_after_reparse",
            }
            reparses.append(row)
            if parse["accepted"]:
                added.append(column)

    public_queries = [{key: value for key, value in query.items() if key != "_entries"}
                      for query in queries]
    return added, {
        "requested_slots": core["implicated_slots"],
        "queries": public_queries,
        "reparse_evidence": reparses,
        "rejections": rejections,
        "added_column_ids": [column.id for column in added],
        "addition_cap": MAX_ADDITIONS_PER_CORE,
        "core_targeted_only": True,
    }


def run_plan(plan: ScenePlan, oracle: LexicalOracle) -> dict[str, Any]:
    active = {slot: [next(column for column in plan.pool[slot] if column.initial)]
              for slot in plan.slots}
    anchor = {slot: columns[0] for slot, columns in active.items()}
    initial_parse = parse_complete_plan(plan, anchor)
    assert initial_parse["accepted"]
    iterations, nogoods, additions = [], [], []
    survivor = None
    final_obstruction = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        model, selects, layout, _ = _build_model(plan, active, nogoods, None)
        solver, status, picked = _solve(model, selects)
        record: dict[str, Any] = {
            "iteration": iteration, "status": solver.status_name(status),
            "active_column_ids": {slot: [c.id for c in cols] for slot, cols in active.items()},
            "packed_sentence_lattice": packed_dag(plan, active),
            "char_offsets": layout,
            "mirror_equalities": len(layout) // 2,
            "nogoods_before_solve": list(nogoods),
        }
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen = {slot: active[slot][index] for slot, index in picked.items()}
            text = render(plan, chosen)
            parse = parse_complete_plan(plan, chosen)
            audit = independent_audit(text)
            central = mechanical_admission_checks(text, min_letters=39, max_letters=240)
            accepted = (parse["accepted"] and audit["letters"] > 38 and
                        audit["two_pointer_exact"] and audit["sha_equal"] and all(central.values()))
            survivor = {"rendered": text, "selected_column_per_slot":
                        {slot: col.id for slot, col in chosen.items()}, "parse": parse,
                        "independent_exact_audit": audit, "central_admission": central,
                        "accepted": accepted,
                        "reader_gate": "closed pending blinded human grammar/meaning/paraphrase ratings"}
            record["selected"] = survivor
            iterations.append(record)
            break
        core = minimum_core(plan, active, nogoods)
        record["unsat_core"] = core
        old_sizes = {slot: len(active[slot]) for slot in core["implicated_slots"]}
        added, response = contextual_columns(plan, active, core, anchor, oracle)
        record["oracle_response"] = response
        iterations.append(record)
        if not added:
            final_obstruction = {"reason": "contextual oracle exhausted for minimum conflict",
                                 "core": core, "active_column_ids": record["active_column_ids"]}
            break
        for column in added:
            active[column.slot].append(column)
            additions.append({"after_iteration": iteration, "slot": column.slot,
                              "column_id": column.id,
                              "reparsed_complete_clause": True})
        nogood = {"id": f"{plan.id}:nogood:{iteration}", "slots": core["implicated_slots"],
                  "old_domain_sizes": old_sizes,
                  "meaning": "at least one implicated slot must select a newly reparsed column"}
        nogoods.append(nogood)
    if survivor is None and final_obstruction is None:
        final_obstruction = {"reason": "eight-iteration cap reached", "nogoods": nogoods}
    final_anchor = {slot: active[slot][0] for slot in plan.slots}
    return {
        "scene_plan": {"id": plan.id, "description": plan.description,
                       "complete_clause_skeletons": [clause.__dict__ for clause in plan.clauses]},
        "initial_complete_control": {"rendered": render(plan, final_anchor),
                                     "parse": initial_parse,
                                     "audit": independent_audit(render(plan, final_anchor))},
        "state_schema": ["scene_plan", "complete_clause_skeletons", "packed_constituent_DAG",
                         "feature_env", "selected_column_per_slot", "char_offsets",
                         "mirror_equalities", "nogoods"],
        "feature_env": {"agreement": "subject-number=finite-verb-number",
                        "valency": "finite transitive verb requires NP object",
                        "attachment": "clause prefix attaches to its finite event",
                        "anaphora": "there resolves to the first-clause scene location",
                        "scene_continuity": plan.location_entity},
        "initial_control_columns": {slot: [column.__dict__ | {"tape": column.tape}
                                           for column in columns]
                                    for slot, columns in plan.pool.items()},
        "oracle_slot_inventory": {
            slot: len(oracle.tries[(plan.id, slot)].entries) for slot in plan.slots
        },
        "position_policy": oracle.provenance["variable_length_policy"],
        "iterations": iterations, "nogoods": nogoods, "added_columns": additions,
        "survivor": survivor, "final_obstruction": final_obstruction,
    }


def run() -> dict[str, Any]:
    plans = build_plans()
    assert len(plans) == 2
    oracle = build_lexical_oracle(plans)
    results = [run_plan(plan, oracle) for plan in plans]
    survivors = [row["survivor"] for row in results if row["survivor"]]
    source = Path(__file__)
    return {
        "experiment_id": EXPERIMENT_ID, "preflight_signature": PREFLIGHT_SIGNATURE,
        "method": "bounded conflict-directed column generation with CP-SAT assumption cores",
        "bounds": {"scene_plans": 2, "max_iterations_per_plan": MAX_ITERATIONS,
                   "max_reparsed_columns_per_core": MAX_ADDITIONS_PER_CORE,
                   "vocabulary": "fixed pre-solve common Brown/WordNet character tries",
                   "max_oracle_entries_per_slot": MAX_ORACLE_ENTRIES_PER_SLOT,
                   "max_query_terminals": MAX_QUERY_TERMINALS},
        "solver": {"engine": "OR-Tools CP-SAT", "ortools_version": __import__("ortools").__version__,
                   "deterministic_workers": 1},
        "lexical_oracle": oracle.provenance,
        "plans": results, "survivors": survivors,
        "acceptance_contract": {"letters_strictly_greater_than": 38,
            "independent_audits": ["two-pointer", "forward/reverse SHA-256"],
            "grammar": ["complete finite clauses", "agreement", "valency", "attachment",
                        "anaphora", "shared scene continuity"],
            "freshness": "all content lemmas distinct",
            "shortcut_gate": "all current central mechanical admission checks",
            "reader_gate": "closed until blinded grammar, meaning, and paraphrase ratings"},
        "provenance": {"host": socket.gethostname(), "python": platform.python_version(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seed_text_used": False, "catalogue_text_used": False,
            "local_r_equals_s_family_used": False, "finished_mirror_units_used": False,
            "fragments_used": False, "post_hoc_repair_used": False,
            "complete_sentence_bank_materialized": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"plans": len(payload["plans"]), "iterations":
                      [len(row["iterations"]) for row in payload["plans"]],
                      "cores": [sum("unsat_core" in item for item in row["iterations"])
                                for row in payload["plans"]],
                      "added_columns": [len(row["added_columns"]) for row in payload["plans"]],
                      "survivors": len(payload["survivors"])}, sort_keys=True))


if __name__ == "__main__":
    main()
