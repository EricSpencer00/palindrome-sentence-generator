"""Freshness-indexed register/stack cycles at a live plural ``s`` residual.

The predecessor adjective cycle at residual ``m`` found only repeated lexical
material.  This bounded successor changes the morphology, not the completed
sentence bank: a productive plural boundary keeps ``s`` live while fresh
typed lemmas are pushed on the left and returned in LIFO order on the right.

Every transition satisfies ``T(x_i) + s = s + reverse(T(y_i))``.  The first
two depths have intact, interpretable readings.  Deeper materializations are
retained as exact mechanical children, but are not promoted as prose when the
fixed return phase can no longer maintain one event domain.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)


ID = "freshness-register-morphology-cycle-20260922"
OUT = ROOT / "runs/freshness-register-morphology-cycle-20260922.json"
RESIDUAL = "s"
CARRIER_CONTENT = frozenset({"trace", "note", "set", "one", "carton"})

# ``reverse(Q) = P + s`` for every row.  The four held-out rows were found by
# the same bounded common-word phrase product as the selected carrier.  They
# are rendered and audited below rather than silently discarded.
CARRIERS = (
    {
        "id": "inspection_carton",
        "p_words": ("no", "trace", "note"),
        "q_words": ("set", "one", "carton"),
        "p_rendered": "No trace. Note:",
        "q_rendered": "Set one carton.",
        "selected": True,
        "continuity": "inspection finding, observation, and packing command share one work scene",
    },
    {
        "id": "race_car",
        "p_words": ("race", "note"),
        "q_words": ("set", "one", "car"),
        "p_rendered": "Race. Note:",
        "q_rendered": "Set one car.",
        "selected": False,
        "continuity": "grammatical commands, but the utensil cycle has no role in the car event",
    },
    {
        "id": "name_man",
        "p_words": ("name", "note"),
        "q_words": ("set", "one", "man"),
        "p_rendered": "Name. Note:",
        "q_rendered": "Set one man.",
        "selected": False,
        "continuity": "bare name command and set-man command lack a recoverable shared task",
    },
    {
        "id": "rise_sir",
        "p_words": ("rise", "note"),
        "q_words": ("set", "one", "sir"),
        "p_rendered": "Rise. Note:",
        "q_rendered": "Set one sir.",
        "selected": False,
        "continuity": "the closing count phrase is grammatical only under an unsuitable title sense",
    },
    {
        "id": "nose_son",
        "p_words": ("nose", "note"),
        "q_words": ("set", "one", "son"),
        "p_rendered": "Nose. Note:",
        "q_rendered": "Set one son.",
        "selected": False,
        "continuity": "dictionary words pass, but the two commands do not form an ordinary event",
    },
)


@dataclass(frozen=True)
class Cycle:
    x: str
    y: str
    x_type: str
    y_type: str
    event_role: str

    def equation(self) -> dict[str, object]:
        left = self.x + RESIDUAL
        right = RESIDUAL + self.y[::-1]
        return {
            "x": self.x,
            "y": self.y,
            "x_type": self.x_type,
            "y_type": self.y_type,
            "event_role": self.event_role,
            "residual": RESIDUAL,
            "left": left,
            "right": right,
            "holds": left == right,
            "residual_nonempty": bool(RESIDUAL),
        }


# Frozen output of the bounded common-word preflight.  All forms are lowercase
# Brown-attested open-class lemmas accepted by the repository lexicon.  The
# reverse orientation of every pair is also in the search domain.
PAIR_FAMILIES = (
    ("sap", "spa"),
    ("sleet", "steel"),
    ("snap", "span"),
    ("snoop", "spoon"),
    ("spot", "stop"),
    ("straw", "swart"),
)

# Inner first.  Extending a witness adds exactly one outer cycle and leaves all
# earlier choices unchanged.  Thus k=3..5 are continuations of the successful
# k=1/k=2 branch, not a new completed-text sweep.
CYCLE_SCHEDULE = (
    Cycle("spoon", "snoop", "N.STEM->N.PL", "V.BASE", "artifact observation"),
    Cycle("spot", "stop", "V.IMP", "V.IMP", "inspection command"),
    Cycle("snap", "span", "V.IMP", "V.TR", "image/span return"),
    Cycle("steel", "sleet", "V.TR", "V.WEATHER", "material/weather return"),
    Cycle("sap", "spa", "V.TR", "V.INTR", "material/leisure return"),
)

# Sentence groups index the 2k surface words between ``Note:`` and the final
# ``Set one carton`` command.  They are punctuation/phase choices only; the
# character tape is wholly determined by the register equations above.
GROUP_WIDTHS = {
    1: (2,),
    2: (2, 1, 1),
    3: (1, 2, 1, 2),
    4: (2, 2, 1, 1, 2),
    5: (2, 1, 2, 1, 1, 2, 1),
}

SYNTAX = {
    1: {
        "direct_status": "intact_interpretable",
        "event_continuity": True,
        "reading": "plural artifact subject plus present-tense observation verb",
        "obstruction": None,
    },
    2: {
        "direct_status": "intact_interpretable",
        "event_continuity": True,
        "reading": "inspection imperative followed by two ordinary imperatives",
        "obstruction": None,
    },
    3: {
        "direct_status": "syntax_register_obstruction",
        "event_continuity": False,
        "reading": "telegraphic imperatives only",
        "obstruction": (
            "the returned lemma 'span' is transitive in the exposed phase; "
            "the exact tape supplies neither a determiner nor an event-compatible "
            "object, so 'stop span' is only a technical shorthand"
        ),
    },
    4: {
        "direct_status": "syntax_register_obstruction",
        "event_continuity": False,
        "reading": "material and weather commands remain tokenizable",
        "obstruction": (
            "the next exact pair steel/sleet can fill verb slots, but it switches "
            "the inspection event to an unlicensed weather return and supplies no "
            "shared argument for the existing span obligation"
        ),
    },
    5: {
        "direct_status": "syntax_register_obstruction",
        "event_continuity": False,
        "reading": "mechanically exact imperative chain",
        "obstruction": (
            "the next exact pair sap/spa preserves the character register, but "
            "the LIFO return exposes leisure-verb 'spa' after the material/weather "
            "chain; no available boundary phase gives it a continuous event role"
        ),
    },
}


def independent_audit(text: str) -> dict[str, object]:
    """Audit without calling the repository palindrome validator."""
    tape = "".join(re.findall(r"[a-z]", text.casefold()))
    mismatch = next(
        (i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]),
        None,
    )
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": mismatch is None and bool(tape),
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def proper_palindromic_word_spans(text: str) -> list[dict[str, object]]:
    """Return proper multiword spans rejected by the boundary mask."""
    words = tokenize(text)
    rows = []
    for start in range(len(words)):
        for stop in range(start + 2, len(words) + 1):
            if start == 0 and stop == len(words):
                continue
            span_tape = "".join(normalize_letters(word) for word in words[start:stop])
            if span_tape == span_tape[::-1]:
                rows.append({"start": start, "stop": stop,
                             "words": list(words[start:stop])})
    return rows


def _surface_words(cycles: tuple[Cycle, ...]) -> tuple[str, ...]:
    left = [cycle.x for cycle in reversed(cycles)]
    left[-1] += RESIDUAL  # boundary migration: spoon + s -> spoons
    right = [cycle.y for cycle in cycles]  # stack pop order: inner to outer
    return tuple((*left, *right))


def _sentences(words: tuple[str, ...], widths: tuple[int, ...]) -> tuple[str, ...]:
    assert sum(widths) == len(words)
    rows = []
    offset = 0
    for width in widths:
        row = " ".join(words[offset:offset + width])
        rows.append(row.capitalize() + ".")
        offset += width
    return tuple(rows)


def _render_with_carrier(depth: int, carrier: dict[str, object]) -> str:
    cycles = CYCLE_SCHEDULE[:depth]
    words = _surface_words(cycles)
    middle = " ".join(_sentences(words, GROUP_WIDTHS[depth]))
    return f"{carrier['p_rendered']} {middle} {carrier['q_rendered']}"


def carrier_equation(carrier: dict[str, object]) -> dict[str, object]:
    p_tape = "".join(carrier["p_words"])
    q_tape = "".join(carrier["q_words"])
    return {
        "p_tape": p_tape,
        "q_tape": q_tape,
        "left": q_tape[::-1],
        "right": p_tape + RESIDUAL,
        "holds": q_tape[::-1] == p_tape + RESIDUAL,
    }


def render_depth(depth: int) -> dict[str, object]:
    if depth not in GROUP_WIDTHS:
        raise ValueError(f"unsupported depth: {depth}")
    cycles = CYCLE_SCHEDULE[:depth]
    carrier = next(row for row in CARRIERS if row["selected"])
    words = _surface_words(cycles)
    rendered = _render_with_carrier(depth, carrier)

    lemmas = tuple(value for cycle in cycles for value in (cycle.x, cycle.y))
    equations = [cycle.equation() for cycle in cycles]
    right_surface = "".join(cycle.y for cycle in cycles)
    left_stack = "".join(cycle.x for cycle in reversed(cycles))
    stacked_left = left_stack + RESIDUAL
    stacked_right = RESIDUAL + right_surface[::-1]
    audit = independent_audit(rendered)
    checks = mechanical_admission_checks(rendered)
    forbidden_spans = proper_palindromic_word_spans(rendered)
    p_tape = "".join(carrier["p_words"])
    q_tape = "".join(carrier["q_words"])
    expected_tape = p_tape + left_stack + RESIDUAL + right_surface + q_tape
    actual_tape = normalize_letters(rendered)
    if actual_tape != expected_tape or not audit["two_pointer_exact"]:
        raise AssertionError(rendered)

    phase_trace = []
    for index, cycle in enumerate(cycles, start=1):
        phase_trace.append({
            "cycle_index_inner_first": index,
            "boundary_phase": (
                "plural_suffix_pending" if index == 1 else
                "imperative_or_modifier_prefix"
            ),
            "residual_before": RESIDUAL,
            "residual_after": RESIDUAL,
            "stack_depth_after_push": index,
            "pushed_y": cycle.y,
            "fresh_lemma_ids": [cycle.x, cycle.y],
        })

    return {
        "depth": depth,
        "rendered": rendered,
        "normalized_tape": actual_tape,
        "surface_words": list(words),
        "cycles_inner_first": equations,
        "stack_pop_order": [cycle.y for cycle in cycles],
        "stacked_equation": {
            "left": stacked_left,
            "right": stacked_right,
            "holds": stacked_left == stacked_right,
        },
        "register_trace": phase_trace,
        "freshness": {
            "lemmas": list(lemmas),
            "all_lemmas_distinct": len(lemmas) == len(set(lemmas)),
            "carrier_content_disjoint": not (set(lemmas) & CARRIER_CONTENT),
        },
        "boundary_mask": {
            "proper_palindromic_multiword_spans": forbidden_spans,
            "passed": not forbidden_spans,
        },
        "independent_audit": audit,
        "mechanical_checks": checks,
        "mechanically_admitted": all(checks.values()),
        "proper_names": False,
        "carrier_id": carrier["id"],
        "carrier_equation": carrier_equation(carrier),
        "reader_certified": False,
        "syntax": SYNTAX[depth],
    }


def run() -> dict[str, object]:
    rows = [render_depth(depth) for depth in range(1, 6)]
    carrier_rows = []
    for carrier in CARRIERS:
        rendered = _render_with_carrier(2, carrier)
        audit = independent_audit(rendered)
        checks = mechanical_admission_checks(rendered)
        spans = proper_palindromic_word_spans(rendered)
        carrier_rows.append({
            "id": carrier["id"],
            "p_words": list(carrier["p_words"]),
            "q_words": list(carrier["q_words"]),
            "equation": carrier_equation(carrier),
            "rendered_k2": rendered,
            "independent_audit": audit,
            "mechanical_checks": checks,
            "mechanically_admitted": all(checks.values()),
            "boundary_mask_passed": not spans,
            "selected": carrier["selected"],
            "continuity_assessment": carrier["continuity"],
        })
    return {
        "experiment_id": ID,
        "method": (
            "freshness-indexed typed register/pushdown cycle with a live plural-s "
            "residual, LIFO returns, and an online complementary-boundary mask"
        ),
        "state": {
            "residual": RESIDUAL,
            "residual_must_remain_nonempty": True,
            "registers": [
                "grammar_phase", "residual_owner", "boundary_phase",
                "fresh_lemma_ids", "content_words", "boundary_mask",
            ],
            "stack_symbol": "(y_lemma, return_phase)",
            "transition_equation": "T(x_i) r = r reverse(T(y_i))",
        },
        "bounded_domain": {
            "source": (
                "frozen lowercase Brown-attested open-class lemmas intersected "
                "with the repository lexicon"
            ),
            "pair_families": [list(pair) for pair in PAIR_FAMILIES],
            "directed_equation_pairs": 2 * len(PAIR_FAMILIES),
            "depths_materialized": [1, 2, 3, 4, 5],
            "same_branch_extension": True,
            "completed_sentence_cartesian_sweep": False,
        },
        "carrier_preflight": {
            "equation": "reverse(T(Q)) = T(P) + s",
            "common_word_pairs_tested": len(carrier_rows),
            "rows": carrier_rows,
            "decision": (
                "retain inspection_carton: every held-out carrier is exact and "
                "mechanically checked, but none connects the utensil/inspection "
                "cycle to its closing event as well as the selected carrier"
            ),
        },
        "children": rows,
        "stats": {
            "rendered_children": len(rows),
            "exact_children": sum(row["independent_audit"]["two_pointer_exact"] for row in rows),
            "mechanically_admitted_children": sum(row["mechanically_admitted"] for row in rows),
            "intact_interpretable_depths": [
                row["depth"] for row in rows
                if row["syntax"]["direct_status"] == "intact_interpretable"
            ],
            "syntax_obstruction_depths": [
                row["depth"] for row in rows
                if row["syntax"]["direct_status"] == "syntax_register_obstruction"
            ],
            "longest_exact_letters": max(row["independent_audit"]["letters"] for row in rows),
        },
        "reader_gate": {
            "status": "closed",
            "reason": (
                "exactness and direct syntactic interpretation are not blinded "
                "reader evidence; depths 3-5 additionally lose event continuity"
            ),
        },
        "obstruction": {
            "first_depth": 3,
            "equation_domain": (
                "after spoon/snoop and spot/stop, the remaining common r=s pairs "
                "expose span, steel/sleet, spa, or swart in the fixed return phase"
            ),
            "register_failure": (
                "those returns require an object/determiner, change the event domain, "
                "or lack a productive finite-verb reading; punctuation cannot add "
                "the missing characters or semantic argument"
            ),
            "next_productive_morphology_operator": (
                "replace bare returned lemmas with typed verb+particle/object frames "
                "inside T(y), such as V+PRON or V+DET+N, then derive T(x) from the "
                "same live-residual equation before rendering"
            ),
        },
        "provenance": {
            "catalogue_text": False,
            "proper_names": False,
            "finished_tape_reversal": False,
            "posthoc_character_repair": False,
            "repeated_content": False,
            "closed_mirrored_units": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def main() -> None:
    data = run()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
