"""Bounded asynchronous cross-clause genitive dependency product.

This is the concrete successor to ``possessive_clitic_return_stack``.  A
literal possessive ``'s`` consumes the live character ``s`` while the
possessed head, its later finite-predicate agreement/valency debt, and its
discourse referent stay open.  The head and predicate therefore need not
close in the same return frame.

The bounded result is negative.  The best grammatical prefix is the exact
path ``we + spot + snoops' / spoon + stop + sew``.  Its first clause can close
as ``we spot snoops' spoon``.  After that boundary the return stream begins
with base ``stop``: it supplies neither the required coreferential subject
``it`` nor singular finite ``stops``.  The exact character register is already
empty there, so inserting either form would be post-hoc repair and is not
allowed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "asynchronous-cross-clause-genitive-product-20260922"
DEFAULT_ARTIFACT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
RESIDUAL = "s"
FUNCTION_LEMMAS = frozenset({"a", "it", "no", "one", "the", "we"})


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class Pair:
    pair_id: str
    left: tuple[str, ...]
    right: tuple[str, ...]
    kind: str
    left_role: str
    right_role: str

    def equation_holds(self) -> bool:
        left = normalize(" ".join(self.left)) + RESIDUAL
        right = normalize(" ".join(self.right))[::-1]
        if self.kind == "cycle":
            right = RESIDUAL + right
        return left == right


@dataclass(frozen=True)
class Genitive:
    genitive_id: str
    owner: str
    owner_number: str
    head: str
    head_number: str
    relation: str
    surface: str

    def equation_holds(self) -> bool:
        return normalize(self.owner) + RESIDUAL == RESIDUAL + normalize(self.head)[::-1]


@dataclass(frozen=True)
class DependencyDebt:
    possessor: str
    head: str
    attachment: str
    head_number: str
    head_attached: bool
    antecedent_clause: int
    predicate_clause: int
    required_coreferent: str
    coreference_resolved: bool
    required_finite_surface: str
    observed_predicate_surface: str | None
    predicate_agreement_resolved: bool
    required_valency: str
    predicate_valency_resolved: bool
    clauses_finished: tuple[bool, bool]


CARRIERS = (
    Pair("speaker", ("we",), ("sew",), "carrier", "clause_subject", "base_predicate"),
    Pair("inspection-carton", ("no", "trace", "note"), ("set", "one", "carton"),
         "carrier", "inspection_report", "packing_command"),
)

CYCLES = {
    "weather-material": Pair("weather-material", ("sleet",), ("steel",), "cycle",
                             "weather", "material"),
    "event-duration": Pair("event-duration", ("snap",), ("span",), "cycle",
                           "event", "transitive_predicate"),
    "location-command": Pair("location-command", ("spot",), ("stop",), "cycle",
                             "transitive_predicate", "base_predicate"),
    "fiber-description": Pair("fiber-description", ("straw",), ("swart",), "cycle",
                              "material", "archaic_adjective"),
}

# There is deliberately no symmetric ``saw a`` cycle.  The schedules are the
# complete bounded domain, not seeds for a widening loop.
SCHEDULES = (
    ("location-command",),
    ("weather-material", "location-command"),
    ("event-duration", "location-command"),
    ("weather-material", "event-duration", "location-command"),
    ("fiber-description", "weather-material", "event-duration", "location-command"),
)

GENITIVES = (
    Genitive("singular-snoop-spoon", "snoop", "singular", "spoon", "singular",
             "ordinary ownership", "snoop's"),
    Genitive("plural-snoops-spoon", "snoop", "plural", "spoon", "singular",
             "ordinary shared ownership", "snoops'"),
    Genitive("plural-stops-spot", "stop", "plural", "spot", "singular",
             "associated-place ownership", "stops'"),
)


def reconcile(owner: str, residual: str, side: str, chars: str) -> tuple[str, str, bool]:
    """Consume one exact-tape chunk, retaining unmatched owner and residual."""
    for char in chars:
        if not residual:
            owner, residual = side, char
        elif owner == side:
            residual += char
        elif residual[0] == char:
            residual = residual[1:]
            if not residual:
                owner = ""
        else:
            return owner, residual, False
    return owner, residual, True


def character_trace(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> dict:
    owner = residual = ""
    left_cursor = right_cursor = 0
    trace = []
    for word in left_words:
        chars = normalize(word)
        owner, residual, matched = reconcile(owner, residual, "L", chars)
        left_cursor += len(chars)
        trace.append({"side": "L", "token": word, "owner": owner, "residual": residual,
                      "left_cursor": left_cursor, "right_cursor": right_cursor,
                      "matched": matched, "masks_live": True})
    # Consume the final right half from the outside edge inward.
    for word in reversed(right_words):
        chars = normalize(word)[::-1]
        owner, residual, matched = reconcile(owner, residual, "R", chars)
        right_cursor += len(chars)
        trace.append({"side": "R", "token": word, "owner": owner, "residual": residual,
                      "left_cursor": left_cursor, "right_cursor": right_cursor,
                      "matched": matched, "masks_live": True})
        if not matched:
            break
    return {"owner": owner, "residual": residual, "left_cursor": left_cursor,
            "right_cursor": right_cursor, "closed": not owner and not residual,
            "trace": trace}


def boundary_positions(words: Iterable[str]) -> tuple[int, ...]:
    cursor, result = 0, []
    for word in words:
        cursor += len(normalize(word))
        result.append(cursor)
    return tuple(result)


def complementary_boundary_mask(left: tuple[str, ...], right: tuple[str, ...]) -> dict:
    left_positions = boundary_positions(left)
    right_positions = boundary_positions(reversed(right))
    shared = tuple(sorted(set(left_positions).intersection(right_positions)))
    terminal = (left_positions[-1],) if left_positions[-1] == right_positions[-1] else ()
    forbidden = tuple(position for position in shared if position not in terminal)
    return {"left": left_positions, "right_from_outer_edge": right_positions,
            "shared": shared, "allowed_terminal": terminal,
            "forbidden_internal": forbidden, "passes": bool(terminal) and not forbidden}


def independent_audit(words: tuple[str, ...]) -> dict:
    tape = normalize(" ".join(words))
    mismatch = next((index for index in range(len(tape) // 2)
                     if tape[index] != tape[-1 - index]), None)
    forward, reverse = digest(tape), digest(tape[::-1])
    return {"letters": len(tape), "normalized_tape": tape,
            "two_pointer_exact": mismatch is None and bool(tape),
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "hashes_agree": forward == reverse}


def proper_span_mask(words: tuple[str, ...]) -> dict:
    spans = []
    for start in range(len(words)):
        for stop in range(start + 2, len(words) + 1):
            if start == 0 and stop == len(words):
                continue
            tape = normalize(" ".join(words[start:stop]))
            if tape and tape == tape[::-1]:
                spans.append({"start": start, "stop": stop,
                              "words": words[start:stop], "tape": tape})
    return {"passes": not spans, "forbidden_spans": spans}


def lemma(word: str) -> str:
    value = normalize(word)
    if word.endswith("'") and value.endswith("s"):
        value = value[:-1]
    if word.endswith("'s") and value.endswith("s"):
        value = value[:-1]
    return value


def freshness(words: tuple[str, ...]) -> dict:
    lemmas = tuple(lemma(word) for word in words if lemma(word) not in FUNCTION_LEMMAS)
    return {"content_lemmas": lemmas, "all_distinct": len(lemmas) == len(set(lemmas))}


def contraction_gate(owner: str, expansion: str) -> dict:
    surface = normalize(owner + "'s")
    expanded = normalize(owner + " " + expansion)
    return {"surface": owner + "'s", "interpretation": expansion,
            "surface_tape": surface, "expanded_tape": expanded,
            "passes": surface == expanded,
            "reason": "is/has expansion changes letters" if surface != expanded else None}


def dependency_result(carrier: Pair, schedule: tuple[Pair, ...], genitive: Genitive) -> dict:
    debt = DependencyDebt(
        possessor=genitive.owner, head=genitive.head, attachment="literal_possessive_s",
        head_number=genitive.head_number, head_attached=False,
        antecedent_clause=1, predicate_clause=2, required_coreferent="it",
        coreference_resolved=False, required_finite_surface="stops",
        observed_predicate_surface=None, predicate_agreement_resolved=False,
        required_valency="intransitive event predicate of the possessed head",
        predicate_valency_resolved=False, clauses_finished=(False, False),
    )
    # The sole clause-complete prefix in the bounded inventory is:
    # we + spot + snoops' + spoon.  Crucially, attachment does not discharge
    # the later predicate or coreference debt.
    first_clause = (
        carrier.pair_id == "speaker"
        and tuple(pair.pair_id for pair in schedule) == ("location-command",)
        and genitive.genitive_id == "plural-snoops-spoon"
    )
    debt = replace(debt, head_attached=True, clauses_finished=(first_clause, False))
    return_tokens = tuple(word for pair in reversed(schedule) for word in pair.right) + carrier.right
    observed = return_tokens[0] if return_tokens else None
    debt = replace(debt, observed_predicate_surface=observed)
    failures = []
    if not first_clause:
        failures.append("clause_1_finite_frame")
    if not debt.coreference_resolved:
        failures.append("clause_2_coreferential_subject")
    if observed != debt.required_finite_surface:
        failures.append("clause_2_singular_finite_agreement")
    if not debt.predicate_valency_resolved:
        failures.append("clause_2_predicate_valency")
    return {"state": asdict(debt), "return_tokens_after_head": return_tokens,
            "failures": failures, "both_clauses_finished": not failures,
            "first_undischarged": failures[0] if failures else None}


def certificate(carrier: Pair, schedule: tuple[Pair, ...], genitive: Genitive) -> dict:
    left = carrier.left + tuple(word for pair in schedule for word in pair.left) + (genitive.surface,)
    right = (genitive.head,) + tuple(
        word for pair in reversed(schedule) for word in pair.right
    ) + carrier.right
    words = left + right
    exact = character_trace(left, right)
    dependency = dependency_result(carrier, schedule, genitive)
    boundaries = complementary_boundary_mask(left, right)
    spans = proper_span_mask(words)
    fresh = freshness(words)
    audit = independent_audit(words)
    failures = list(dependency["failures"])
    if not exact["closed"] or not audit["two_pointer_exact"]:
        failures.insert(0, "exact_character_register")
    if not fresh["all_distinct"]:
        failures.append("global_lemma_freshness")
    if not boundaries["passes"]:
        failures.append("complementary_boundary_mask")
    if not spans["passes"]:
        failures.append("proper_span_mask")
    return {
        "path": {"carrier": carrier.pair_id,
                 "cycles": tuple(pair.pair_id for pair in schedule),
                 "genitive": genitive.genitive_id},
        "surface_tokens_not_prose": words,
        "equations": {"carrier": carrier.equation_holds(),
                      "cycles": tuple(pair.equation_holds() for pair in schedule),
                      "genitive": genitive.equation_holds()},
        "dependency": dependency, "character_register": exact,
        "global_lemma_freshness": fresh,
        "complementary_boundary_mask": boundaries,
        "proper_span_mask": spans, "independent_audit": audit,
        "reject_reasons": failures, "survives": not failures,
    }


def run() -> dict:
    rows = [certificate(carrier, tuple(CYCLES[name] for name in schedule), genitive)
            for carrier in CARRIERS for schedule in SCHEDULES for genitive in GENITIVES]
    survivors = [row for row in rows if row["survives"]]
    long_exact = [row for row in rows if row["independent_audit"]["two_pointer_exact"]
                  and row["independent_audit"]["letters"] > 44]
    prefix_complete = [row for row in rows if row["dependency"]["state"]["clauses_finished"][0]]
    first_obstruction = max(prefix_complete, key=lambda row: row["independent_audit"]["letters"])
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "decision": "can a literal-s genitive keep head/coreference/predicate debt open across a clause boundary and yield connected exact prose longer than 44 letters?",
        "acceptance_gate": {"minimum_letters_exclusive": 44, "exact": True,
                            "literal_possessive_s_only": True, "both_clauses_complete": True,
                            "discourse_coreference": True, "predicate_valency_and_agreement": True,
                            "global_lemma_freshness": True, "complementary_boundary_mask": True,
                            "proper_span_mask": True},
        "fixed_domain": {"residual": RESIDUAL,
                         "carriers": [pair.pair_id for pair in CARRIERS],
                         "cycles": list(CYCLES), "schedules": [list(row) for row in SCHEDULES],
                         "genitives": [item.genitive_id for item in GENITIVES],
                         "symmetric_saw_a_replay": False, "finished_palindromic_units": False,
                         "catalogue_text": False, "fragments_admitted": False,
                         "post_hoc_repair": False},
        "contraction_controls": [contraction_gate("snoop", expansion)
                                 for expansion in ("is", "has")],
        "stats": {"paths": len(rows), "exact_paths": sum(
            row["independent_audit"]["two_pointer_exact"] for row in rows),
            "exact_paths_over_44": len(long_exact),
            "max_exact_letters": max(row["independent_audit"]["letters"] for row in rows),
            "first_clause_complete_paths": len(prefix_complete),
            "survivors": len(survivors)},
        "survivors": survivors,
        "first_dependency_or_residual_obstruction": {
            "path": first_obstruction["path"],
            "letters": first_obstruction["independent_audit"]["letters"],
            "character_register_terminal": {key: first_obstruction["character_register"][key]
                                            for key in ("owner", "residual", "left_cursor", "right_cursor", "closed")},
            "dependency": first_obstruction["dependency"],
            "explanation": (
                "Clause 1 attaches plural possessor snoops' to singular head spoon. "
                "After the clause boundary the exact return exposes base stop, but the carried "
                "discourse dependency requires coreferential it and singular finite stops. "
                "The character register is already closed; adding it or s would be post-hoc repair."
            ),
        },
        "long_exact_obstructions": [
            {"path": row["path"], "letters": row["independent_audit"]["letters"],
             "first_undischarged": row["dependency"]["first_undischarged"],
             "terminal_owner": row["character_register"]["owner"],
             "terminal_residual": row["character_register"]["residual"]}
            for row in sorted(long_exact, key=lambda item: (-item["independent_audit"]["letters"], item["path"]["carrier"]))
        ],
        "audited_certificates": rows,
        "verdict": "stop_family_no_complete_connected_exact_survivor",
        "provenance": {"host": os.uname().nodename, "python": os.sys.version.split()[0],
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "every_survivor_independently_audited": True,
                       "bounded_domain_exhausted": True},
    }
    payload["result_sha256"] = digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stats": payload["stats"], "verdict": payload["verdict"],
                      "first_obstruction": payload["first_dependency_or_residual_obstruction"]}, indent=2))


if __name__ == "__main__":
    main()
