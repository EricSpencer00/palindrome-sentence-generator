"""Repair the first asynchronous-genitive dependency inside the live equation.

This bounded successor changes exactly one operator from
``asynchronous_cross_clause_genitive_product_20260922``: clause two is
committed with the coreferential subject ``it`` and the finite 3sg surface
``stops`` before character matching may close.  The productive ``s``
exponents may be assigned only to a singular possessive, a plural possessor,
or a present-tense 3sg verb, and only licensed assignments enter the product.

The result is a bounded obstruction, not a repaired palindrome.  Once outer
``sew`` has matched ``we``, reverse consumption of finite ``stops`` exposes
``s`` where the left frontier requires the ``p`` of ``spot``.  Possessive and
plural boundaries lie deeper in the tape, so moving either of their licensed
``s`` exponents cannot change this cursor.  No word, affix, or punctuation is
inserted after that mismatch.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.asynchronous_cross_clause_genitive_product_20260922 import (
    CARRIERS,
    CYCLES,
    GENITIVES,
    SCHEDULES,
    complementary_boundary_mask,
    digest,
    independent_audit,
    lemma,
    normalize,
    proper_span_mask,
)


EXPERIMENT_ID = "anaphoric-inflection-dependency-product-20260922"
DEFAULT_ARTIFACT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"


@dataclass(frozen=True)
class MorphologyTransition:
    """Two productive-s exponents assigned to licensed surface boundaries."""

    transition_id: str
    possessor_number: str
    possessor_boundary: str
    genitive_exponent: str
    plural_exponent: str
    finite_exponent: str
    head_number: str = "singular"
    pronoun: str = "it"
    verb_lemma: str = "stop"

    @property
    def finite_surface(self) -> str:
        return self.verb_lemma + self.finite_exponent

    @property
    def licensed(self) -> bool:
        singular_possessive = (
            self.possessor_number == "singular"
            and self.possessor_boundary == "possessive"
            and self.genitive_exponent == "s"
            and not self.plural_exponent
        )
        plural_possessive = (
            self.possessor_number == "plural"
            and self.possessor_boundary == "plural"
            and not self.genitive_exponent
            and self.plural_exponent == "s"
        )
        singular_anaphora = self.head_number == "singular" and self.pronoun == "it"
        finite_3sg = self.finite_exponent == "s" and self.finite_surface == "stops"
        # A plural possessive uses the ordinary plural s plus apostrophe; it
        # does not receive a second genitive s.  Both licensed realizations
        # still owe the independent 3sg s on clause two's finite predicate.
        return (singular_possessive or plural_possessive) and singular_anaphora and finite_3sg

    @property
    def productive_s_sites(self) -> tuple[str, ...]:
        sites = []
        if self.genitive_exponent:
            sites.append("possessive")
        if self.plural_exponent:
            sites.append("plural")
        if self.finite_exponent:
            sites.append("third_person_singular")
        return tuple(sites)


MORPHOLOGY_TRANSITIONS = (
    MorphologyTransition(
        "singular-possessive-plus-3sg", "singular", "possessive", "s", "", "s"
    ),
    MorphologyTransition(
        "plural-possessive-plus-3sg", "plural", "plural", "", "s", "s"
    ),
)

# These assignments are measured controls and never enter surface expansion.
REJECTED_MIGRATIONS = (
    MorphologyTransition(
        "bare-possessor", "singular", "none", "", "", "s"
    ),
    MorphologyTransition(
        "plural-double-genitive", "plural", "plural", "s", "s", "s"
    ),
    MorphologyTransition(
        "base-verb-with-it", "plural", "plural", "", "s", ""
    ),
)


def possessor_surface(lemma: str, transition: MorphologyTransition) -> str:
    assert transition.licensed
    if transition.possessor_boundary == "possessive":
        return lemma + "'s"
    return lemma + transition.plural_exponent + "'"


def stream_character_equation(
    left_words: tuple[str, ...], right_words: tuple[str, ...]
) -> dict:
    """Consume the two committed frontiers and stop at the first mismatch.

    ``right_words`` is stored in ordinary reading order.  Exact comparison
    consumes it from the outside edge, so both token order and character order
    are reversed.  Cursors count successfully consumed characters; the
    obstruction records the next attempted cursor separately.
    """
    left_tape = normalize(" ".join(left_words))
    owner = "L" if left_tape else ""
    residual = left_tape
    trace = []
    left_cursor = len(left_tape)
    right_cursor = 0
    mismatch = None
    for token_index in range(len(right_words) - 1, -1, -1):
        token = right_words[token_index]
        token_tape = normalize(token)
        for reverse_index, char in enumerate(reversed(token_tape)):
            expected = residual[0] if residual else None
            step = {
                "side": "R",
                "token": token,
                "ordinary_token_index": token_index,
                "reverse_character_index": reverse_index,
                "surface_character_index": len(token_tape) - 1 - reverse_index,
                "left_cursor": left_cursor,
                "right_cursor_before": right_cursor,
                "attempted_right_cursor": right_cursor + 1,
                "owner": owner,
                "residual_before": residual,
                "expected_character": expected,
                "observed_character": char,
                "matched": expected == char,
                "masks_live": True,
            }
            trace.append(step)
            if expected != char:
                mismatch = step
                break
            residual = residual[1:]
            right_cursor += 1
            if not residual:
                owner = ""
        if mismatch:
            break
    closed = mismatch is None and not owner and not residual
    if mismatch is None and (owner or residual):
        mismatch = {
            "reason": "right_frontier_exhausted",
            "left_cursor": left_cursor,
            "right_cursor_before": right_cursor,
            "attempted_right_cursor": None,
            "owner": owner,
            "residual_before": residual,
            "expected_character": residual[0] if residual else None,
            "observed_character": None,
        }
    return {
        "owner": owner,
        "residual": residual,
        "left_cursor": left_cursor,
        "right_cursor": right_cursor,
        "closed": closed,
        "first_mismatch": mismatch,
        "trace": trace,
    }


def render_surface(words: tuple[str, ...], clause_one_stop: int) -> str:
    clause_one = " ".join(words[:clause_one_stop]).capitalize() + "."
    clause_two = " ".join(words[clause_one_stop:clause_one_stop + 2]).capitalize() + "."
    remainder = words[clause_one_stop + 2:]
    if remainder:
        return f"{clause_one} {clause_two} {' '.join(remainder).capitalize()}."
    return f"{clause_one} {clause_two}"


def dependency_state(
    genitive, transition: MorphologyTransition, return_tail: tuple[str, ...]
) -> dict:
    coreference = transition.pronoun == "it" and transition.head_number == "singular"
    agreement = transition.finite_surface == "stops" and coreference
    valency = agreement  # stop is licensed as an intransitive finite predicate
    return {
        "possessor": genitive.owner,
        "possessor_number": transition.possessor_number,
        "possessor_boundary": transition.possessor_boundary,
        "head": genitive.head,
        "head_number": transition.head_number,
        "head_attachment": "literal_genitive",
        "head_attached": True,
        "antecedent_clause": 1,
        "predicate_clause": 2,
        "required_coreferent": "it",
        "observed_coreferent": transition.pronoun,
        "coreference_resolved": coreference,
        "verb_lemma": transition.verb_lemma,
        "required_finite_surface": "stops",
        "observed_finite_surface": transition.finite_surface,
        "predicate_agreement_resolved": agreement,
        "required_valency": "intransitive event predicate of the possessed head",
        "predicate_valency_resolved": valency,
        "productive_s_sites": transition.productive_s_sites,
        "productive_s_count": len(transition.productive_s_sites),
        "clause_two_tokens_committed_before_equation": (
            transition.pronoun,
            transition.finite_surface,
        ),
        "return_tail_after_clause_two": return_tail,
        "clauses_finished": (False, coreference and agreement and valency),
    }


def morphology_aware_freshness(
    words: tuple[str, ...], possessor_index: int, finite_index: int,
    genitive, transition: MorphologyTransition,
) -> dict:
    """Lemmatize the two migrated-s surfaces before testing freshness."""
    content_lemmas = []
    for index, word in enumerate(words):
        normalized = normalize(word)
        if normalized in {"a", "it", "no", "one", "the", "we"}:
            continue
        if index == possessor_index:
            content_lemmas.append(genitive.owner)
        elif index == finite_index:
            content_lemmas.append(transition.verb_lemma)
        else:
            content_lemmas.append(lemma(word))
    return {
        "content_lemmas": tuple(content_lemmas),
        "all_distinct": len(content_lemmas) == len(set(content_lemmas)),
        "morphological_surfaces_resolved": {
            words[possessor_index]: genitive.owner,
            words[finite_index]: transition.verb_lemma,
        },
    }


def certificate(carrier, schedule, genitive, transition: MorphologyTransition) -> dict:
    assert transition.licensed
    assert schedule and schedule[-1].pair_id == "location-command"
    left_prefix = carrier.left + tuple(word for pair in schedule for word in pair.left)
    possessor = possessor_surface(genitive.owner, transition)
    left = left_prefix + (possessor,)
    old_returns = tuple(word for pair in reversed(schedule) for word in pair.right) + carrier.right
    assert old_returns[0] == "stop"
    # The old base return is replaced, not retained and repaired later.
    return_tail = old_returns[1:]
    right = (genitive.head, transition.pronoun, transition.finite_surface) + return_tail
    words = left + right
    clause_one_stop = len(left) + 1
    rendered = render_surface(words, clause_one_stop)
    dependency = dependency_state(genitive, transition, return_tail)
    first_clause_complete = (
        carrier.pair_id == "speaker"
        and tuple(pair.pair_id for pair in schedule) == ("location-command",)
        and genitive.owner == "snoop"
        and genitive.head == "spoon"
    )
    dependency["clauses_finished"] = (first_clause_complete, dependency["clauses_finished"][1])
    dependency["both_required_clauses_finished"] = all(dependency["clauses_finished"])
    dependency["connected_discourse"] = (
        dependency["both_required_clauses_finished"] and not return_tail
    )

    register = stream_character_equation(left, right)
    audit = independent_audit(words)
    boundaries = complementary_boundary_mask(left, right)
    spans = proper_span_mask(words)
    possessor_index = len(left) - 1
    finite_index = len(left) + 2
    fresh = morphology_aware_freshness(
        words, possessor_index, finite_index, genitive, transition
    )
    reject_reasons = []
    if not register["closed"] or not audit["two_pointer_exact"]:
        reject_reasons.append("exact_character_register")
    if not dependency["both_required_clauses_finished"]:
        reject_reasons.append("two_complete_clauses")
    if not dependency["connected_discourse"]:
        reject_reasons.append("connected_discourse")
    if not fresh["all_distinct"]:
        reject_reasons.append("global_lemma_freshness")
    if not boundaries["passes"]:
        reject_reasons.append("complementary_boundary_mask")
    if not spans["passes"]:
        reject_reasons.append("proper_span_mask")
    if audit["letters"] <= 44:
        reject_reasons.append("length_over_44")
    return {
        "path": {
            "carrier": carrier.pair_id,
            "cycles": tuple(pair.pair_id for pair in schedule),
            "genitive": genitive.genitive_id,
            "morphology_transition": transition.transition_id,
        },
        "rendered": rendered,
        "surface_tokens": words,
        "left_frontier": left,
        "right_frontier": right,
        "dependency": dependency,
        "morphology": asdict(transition) | {
            "finite_surface": transition.finite_surface,
            "productive_s_sites": transition.productive_s_sites,
            "licensed": transition.licensed,
        },
        "character_register": register,
        "global_lemma_freshness": fresh,
        "complementary_boundary_mask": boundaries,
        "proper_span_mask": spans,
        "independent_audit": audit,
        "reject_reasons": reject_reasons,
        "survives": not reject_reasons,
    }


def matching_transition(genitive) -> MorphologyTransition:
    if genitive.owner_number == "singular":
        return MORPHOLOGY_TRANSITIONS[0]
    return MORPHOLOGY_TRANSITIONS[1]


def run() -> dict:
    rows = [
        certificate(
            carrier,
            tuple(CYCLES[name] for name in schedule),
            genitive,
            matching_transition(genitive),
        )
        for carrier in CARRIERS
        for schedule in SCHEDULES
        for genitive in GENITIVES
    ]
    survivors = [row for row in rows if row["survives"]]
    grammatical_prefix_rows = [
        row for row in rows
        if row["dependency"]["both_required_clauses_finished"]
        and row["path"]["genitive"] == "plural-snoops-spoon"
    ]
    first = min(
        grammatical_prefix_rows,
        key=lambda row: (
            len(row["path"]["cycles"]),
            row["path"]["carrier"],
        ),
    )
    mismatch = first["character_register"]["first_mismatch"]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "decision": (
            "does committing coreferential it and agreement-bearing stops inside the "
            "asynchronous genitive equation yield complete connected exact prose over 44 letters?"
        ),
        "acceptance_gate": {
            "minimum_letters_exclusive": 44,
            "exact": True,
            "literal_genitive_head_attachment": True,
            "clause_two_explicit_coreferential_it": True,
            "clause_two_finite_3sg_surface": "stops",
            "predicate_valency_and_agreement": True,
            "complete_connected_discourse": True,
            "global_lemma_freshness": True,
            "complementary_boundary_mask": True,
            "proper_span_mask": True,
        },
        "fixed_domain": {
            "predecessor": "asynchronous-cross-clause-genitive-product-20260922",
            "carriers": [pair.pair_id for pair in CARRIERS],
            "cycles": list(CYCLES),
            "schedules": [list(row) for row in SCHEDULES],
            "genitives": [item.genitive_id for item in GENITIVES],
            "lexical_widening": False,
            "changed_operator_only": "clause_2_anaphora_and_finite_inflection",
            "post_hoc_insertion": False,
            "symmetric_replay": False,
            "fragments_admitted": False,
            "catalogue_text": False,
            "finished_palindromic_units": False,
        },
        "morphology_domain": {
            "productive_residual": "s",
            "licensed_transitions": [
                asdict(item) | {
                    "productive_s_sites": item.productive_s_sites,
                    "finite_surface": item.finite_surface,
                }
                for item in MORPHOLOGY_TRANSITIONS
            ],
            "rejected_unlicensed_controls": [
                asdict(item) | {"licensed": item.licensed}
                for item in REJECTED_MIGRATIONS
            ],
        },
        "stats": {
            "paths": len(rows),
            "clause_two_committed_paths": sum(
                row["dependency"]["clauses_finished"][1] for row in rows
            ),
            "both_required_clauses_finished_paths": len(grammatical_prefix_rows),
            "equation_closures": sum(row["character_register"]["closed"] for row in rows),
            "exact_paths_over_44": sum(
                row["independent_audit"]["two_pointer_exact"]
                and row["independent_audit"]["letters"] > 44
                for row in rows
            ),
            "max_letters": max(row["independent_audit"]["letters"] for row in rows),
            "survivors": len(survivors),
        },
        "survivors": survivors,
        "first_pronoun_inflection_cursor_obstruction": {
            "path": first["path"],
            "rendered": first["rendered"],
            "dependency": first["dependency"],
            "morphology": first["morphology"],
            "cursor": mismatch,
            "terminal_register": {
                key: first["character_register"][key]
                for key in ("owner", "residual", "left_cursor", "right_cursor", "closed")
            },
            "explanation": (
                "Outer sew consumes reverse(wes) against initial we+s of spot. "
                "The next outside character is the finite 3sg exponent s of stops, "
                "but the live left residual requires p. Possessive/plural s boundaries "
                "occur deeper and cannot migrate to this cursor under English syntax."
            ),
        },
        "audited_candidates": rows,
        "verdict": "bounded_pronoun_inflection_cursor_obstruction",
        "provenance": {
            "host": os.uname().nodename,
            "python": os.sys.version.split()[0],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "every_survivor_rendered_and_audited": True,
            "all_candidates_audited": True,
            "bounded_domain_exhausted": True,
        },
    }
    payload["result_sha256"] = digest(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "stats": payload["stats"],
        "verdict": payload["verdict"],
        "first_obstruction": payload["first_pronoun_inflection_cursor_obstruction"],
    }, indent=2))


if __name__ == "__main__":
    main()
