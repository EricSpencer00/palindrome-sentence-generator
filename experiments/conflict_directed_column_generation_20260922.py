"""Conflict-directed column generation over complete semantic clauses.

Two finite scene plans are fixed before solving.  Each plan starts with one
column per constituent.  CP-SAT selects columns and enforces every mirrored
character equality.  On UNSAT, an assumption core is reduced to a
cardinality-minimum conflict, converted to a constituent nogood, and sent to
the contextual oracle.  The oracle can expose only predeclared alternatives
for implicated slots, and every exposed column is reparsed in its complete
clause before the next solve.

This is deliberately a bounded obstruction experiment, not a sentence-bank
sweep: there are exactly two plans, at most eight solves per plan, and at most
two reparsed columns per conflict response.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import platform
from pathlib import Path
import re
import socket
from typing import Any

from ortools.sat.python import cp_model

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


def _col(plan: str, slot: str, text: str, category: str, *, initial: bool = False,
         number: str = "na", valency: str = "na", semantic_type: str = "na",
         lemma: str = "") -> Column:
    stem = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")
    return Column(f"{plan}:{slot}:{stem}", slot, text, category, number,
                  valency, semantic_type, lemma, initial)


def build_plans() -> tuple[ScenePlan, ScenePlan]:
    """Return the entire fixed oracle inventory; it is never widened at run time."""
    plans = []

    plan = "orbital-meal"
    specs = {
        "opening": (
            _col(plan, "opening", "In orbit", "locative", initial=True, semantic_type="place", lemma="orbit"),
            _col(plan, "opening", "In space", "locative", semantic_type="place", lemma="space"),
        ),
        "observer": (
            _col(plan, "observer", "the pilot", "np", initial=True, number="sg", semantic_type="human", lemma="pilot"),
            _col(plan, "observer", "the scout", "np", number="sg", semantic_type="human", lemma="scout"),
        ),
        "observe": (
            _col(plan, "observe", "charts", "finite_verb", initial=True, number="sg", valency="transitive", lemma="chart"),
            _col(plan, "observe", "tracks", "finite_verb", number="sg", valency="transitive", lemma="track"),
        ),
        "sky_object": (
            _col(plan, "sky_object", "the comet", "np", initial=True, number="sg", semantic_type="celestial", lemma="comet"),
        ),
        "anaphoric_link": (
            _col(plan, "anaphoric_link", "Later there", "discourse_anaphor", initial=True, semantic_type="place", lemma=""),
        ),
        "server": (
            _col(plan, "server", "the cook", "np", initial=True, number="sg", semantic_type="human", lemma="cook"),
            _col(plan, "server", "the aide", "np", number="sg", semantic_type="human", lemma="aide"),
        ),
        "serve": (
            _col(plan, "serve", "serves", "finite_verb", initial=True, number="sg", valency="transitive", lemma="serve"),
            _col(plan, "serve", "plates", "finite_verb", number="sg", valency="transitive", lemma="plate"),
        ),
        "meal": (
            _col(plan, "meal", "zucchini", "np", initial=True, number="sg", semantic_type="food", lemma="zucchini"),
            _col(plan, "meal", "macaroni", "np", number="sg", semantic_type="food", lemma="macaroni"),
            _col(plan, "meal", "a risotto", "np", number="sg", semantic_type="food", lemma="risotto"),
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
            _col(plan, "camper", "the camper", "np", number="sg", semantic_type="human", lemma="camper"),
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
            _col(plan, "server", "the aide", "np", number="sg", semantic_type="human", lemma="aide"),
        ),
        "serve": (
            _col(plan, "serve", "serves", "finite_verb", initial=True, number="sg", valency="transitive", lemma="serve"),
            _col(plan, "serve", "plates", "finite_verb", number="sg", valency="transitive", lemma="plate"),
        ),
        "meal": (
            _col(plan, "meal", "salad", "np", initial=True, number="sg", semantic_type="food", lemma="salad"),
            _col(plan, "meal", "pasta", "np", number="sg", semantic_type="food", lemma="pasta"),
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
              "tape_length": len(active[slot][0].tape)} for slot in plan.slots]
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
        lengths = {len(column.tape) for column in plan.pool[slot]}
        if len(lengths) != 1:
            raise AssertionError(f"variable-length column pool for {slot}: {lengths}")
        for local in range(next(iter(lengths))):
            layout.append({"position": cursor, "slot": slot, "local_offset": local})
            cursor += 1
    return layout


def _build_model(plan: ScenePlan, active: dict[str, list[Column]], nogoods: list[dict[str, Any]],
                 equalities: tuple[int, ...] | None, assumptions: bool = False):
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
                       anchor: dict[str, Column]) -> tuple[list[Column], list[dict[str, Any]]]:
    added, evidence = [], []
    for slot in core["implicated_slots"]:
        active_ids = {column.id for column in active[slot]}
        for column in plan.pool[slot]:
            if column.id in active_ids or len(added) >= MAX_ADDITIONS_PER_CORE:
                continue
            trial = dict(anchor)
            trial[slot] = column
            parse = parse_complete_plan(plan, trial)
            clause = next(row for row in parse["clauses"] if slot in next(
                item.slots for item in plan.clauses if item.id == row["clause_id"]))
            row = {"column_id": column.id, "slot": slot, "surface": column.text,
                   "full_clause_surface": clause["surface"], "full_plan_accepted": parse["accepted"],
                   "checks": parse["checks"]}
            evidence.append(row)
            if parse["accepted"]:
                added.append(column)
            if len(added) >= MAX_ADDITIONS_PER_CORE:
                break
        if len(added) >= MAX_ADDITIONS_PER_CORE:
            break
    return added, evidence


def run_plan(plan: ScenePlan) -> dict[str, Any]:
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
        added, reparse = contextual_columns(plan, active, core, anchor)
        record["oracle_response"] = {"requested_slots": core["implicated_slots"],
                                     "reparse_evidence": reparse,
                                     "added_column_ids": [column.id for column in added]}
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
        "fixed_oracle_pool": {slot: [column.__dict__ | {"tape": column.tape}
                                     for column in columns]
                              for slot, columns in plan.pool.items()},
        "iterations": iterations, "nogoods": nogoods, "added_columns": additions,
        "survivor": survivor, "final_obstruction": final_obstruction,
    }


def run() -> dict[str, Any]:
    plans = build_plans()
    assert len(plans) == 2
    results = [run_plan(plan) for plan in plans]
    survivors = [row["survivor"] for row in results if row["survivor"]]
    source = Path(__file__)
    return {
        "experiment_id": EXPERIMENT_ID, "preflight_signature": PREFLIGHT_SIGNATURE,
        "method": "bounded conflict-directed column generation with CP-SAT assumption cores",
        "bounds": {"scene_plans": 2, "max_iterations_per_plan": MAX_ITERATIONS,
                   "max_reparsed_columns_per_core": MAX_ADDITIONS_PER_CORE,
                   "vocabulary": "fixed common-word pools declared in source"},
        "solver": {"engine": "OR-Tools CP-SAT", "ortools_version": __import__("ortools").__version__,
                   "deterministic_workers": 1},
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
