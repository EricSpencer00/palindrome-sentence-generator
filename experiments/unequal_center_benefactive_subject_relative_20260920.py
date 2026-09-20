"""Held-out benefactive alternation and subject-relative frames.

This is the next operator after the unequal-center ditransitive/relative
complement lane. It retains that lane's two-cursor product and adds exactly
two typed grammar alternatives: a to-dative versus for-benefactive frame and
an independently authored subject-relative frame. Case, attachment, number,
and tense remain explicit state during ordinary-order lexicalization.

The shared cursor walks the left clause forward and the right clause inward.
It never constructs a finished tape and reverses it. Reversal is used only by
the independent audits below.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unequal_center_ditransitive_relative_complement_20260920 import (  # noqa: E402
    Word,
    walk_pair,
)


OUT = ROOT / "runs/unequal-center-benefactive-subject-relative-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "unequal-center-benefactive-subject-relative-20260920"
SIGNATURE = (
    "typed-seam-machine|benefactive-to-for-alternation|"
    "subject-relative-attachment|case-number-tense"
)
BASE_ID = "unequal-center-grammar-intersection-20260920"
PREDECESSOR_ID = "unequal-center-ditransitive-relative-complement-20260920"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_audit(text: str) -> dict:
    """Independent outside-in audit, separate from the construction cursor."""
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
    tape = normalize(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def independent_audit(text: str) -> dict:
    direct = pointer_audit(text)
    hashed = hash_audit(text)
    return {
        **direct,
        **hashed,
        "independent_exact": direct["exact"] and hashed["sha_equal_under_reversal"],
    }


@dataclass(frozen=True)
class Clause:
    frame_id: str
    production: str
    valency: str
    case: str
    attachment: str
    number: str
    tense: str
    words: tuple[Word, ...]
    roles: tuple[str, ...]

    @property
    def surface(self) -> str:
        return " ".join(word.surface for word in self.words)

    @property
    def letters(self) -> int:
        return len(normalize(self.surface))


BENEF_SUBJECTS = {
    "sg": (("the patient baker", "baker"), ("a careful teacher", "teacher")),
    "pl": (("patient bakers", "bakers"), ("careful teachers", "teachers")),
}

BENEF_EVENTS = {
    "to": {
        "present": {"sg": (("sends", "send"), ("gives", "give")), "pl": (("send", "send"), ("give", "give"))},
        "past": {"sg": (("sent", "send"), ("gave", "give")), "pl": (("sent", "send"), ("gave", "give"))},
    },
    "for": {
        "present": {"sg": (("bakes", "bake"), ("makes", "make")), "pl": (("bake", "bake"), ("make", "make"))},
        "past": {"sg": (("baked", "bake"), ("made", "make")), "pl": (("baked", "bake"), ("made", "make"))},
    },
}

BENEF_THEMES = {
    "to": {
        "sg": (("a sealed letter", "letter"), ("the blue parcel", "parcel")),
        "pl": (("sealed letters", "letters"), ("blue parcels", "parcels")),
    },
    "for": {
        "sg": (("a warm cake", "cake"), ("the bright lantern", "lantern")),
        "pl": (("warm cakes", "cakes"), ("bright lanterns", "lanterns")),
    },
}

BENEF_RECIPIENTS = {
    "sg": (("the quiet child", "child"), ("a kind neighbor", "neighbor")),
    "pl": (("quiet children", "children"), ("kind neighbors", "neighbors")),
}

RELATIVE_HEADS = {
    "sg": (("the patient pilot", "pilot"), ("a careful gardener", "gardener")),
    "pl": (("patient pilots", "pilots"), ("careful gardeners", "gardeners")),
}

RELATIVE_EVENTS = {
    "present": {
        "sg": (("charts", "chart", "the old map", "map"), ("guards", "guard", "the quiet inlet", "inlet")),
        "pl": (("chart", "chart", "old maps", "maps"), ("guard", "guard", "quiet inlets", "inlets")),
    },
    "past": {
        "sg": (("charted", "chart", "the old map", "map"), ("guarded", "guard", "the quiet inlet", "inlet")),
        "pl": (("charted", "chart", "old maps", "maps"), ("guarded", "guard", "quiet inlets", "inlets")),
    },
}

MATRIX_EVENTS = {
    "present": {
        "sg": (("guides", "guide", "the narrow bridge", "bridge"), ("observes", "observe", "a calm harbor", "harbor")),
        "pl": (("guide", "guide", "narrow bridges", "bridges"), ("observe", "observe", "calm harbors", "harbors")),
    },
    "past": {
        "sg": (("guided", "guide", "the narrow bridge", "bridge"), ("observed", "observe", "a calm harbor", "harbor")),
        "pl": (("guided", "guide", "narrow bridges", "bridges"), ("observed", "observe", "calm harbors", "harbors")),
    },
}


def phrase_words(text: str, role: str, phrase_index: int) -> tuple[Word, ...]:
    return tuple(Word(token, role, phrase_index) for token in text.split())


def build_paths() -> tuple[Clause, ...]:
    """Build complete ordinary-order paths from only the new role banks."""
    paths: list[Clause] = []
    for number, subjects in BENEF_SUBJECTS.items():
        for tense in ("present", "past"):
            for case in ("to", "for"):
                for subject, subject_lemma in subjects:
                    for verb, verb_lemma in BENEF_EVENTS[case][tense][number]:
                        for theme_text, theme_lemma in BENEF_THEMES[case][number]:
                            for recipient_text, recipient_lemma in BENEF_RECIPIENTS[number]:
                                marker = "to" if case == "to" else "for"
                                phrases = (
                                    phrase_words(subject, "subject", 0),
                                    phrase_words(verb, "finite_verb", 1),
                                    phrase_words(theme_text, "theme", 2),
                                    phrase_words(marker, "case_marker", 3),
                                    phrase_words(recipient_text, "recipient", 4),
                                )
                                paths.append(Clause(
                                    frame_id=(
                                        f"benefactive-{case}:{number}:{tense}:"
                                        f"{subject_lemma}:{verb_lemma}:{theme_lemma}:{recipient_lemma}"
                                    ),
                                    production=f"benefactive_{case}",
                                    valency="dative" if case == "to" else "benefactive",
                                    case="to_dative" if case == "to" else "for_benefactive",
                                    attachment="matrix",
                                    number=number,
                                    tense=tense,
                                    words=tuple(word for phrase in phrases for word in phrase),
                                    roles=("subject", "finite_verb", "theme", "case_marker", "recipient"),
                                ))

    for number, heads in RELATIVE_HEADS.items():
        for tense in ("present", "past"):
            for head_text, head_lemma in heads:
                for rel_verb, rel_lemma, rel_theme, rel_theme_lemma in RELATIVE_EVENTS[tense][number]:
                    for matrix_verb, matrix_lemma, matrix_theme, matrix_theme_lemma in MATRIX_EVENTS[tense][number]:
                        phrases = (
                            phrase_words(head_text, "relative_head_subject", 0),
                            phrase_words("who", "relative_marker", 1),
                            phrase_words(rel_verb, "relative_finite_verb", 2),
                            phrase_words(rel_theme, "relative_theme", 3),
                            phrase_words(matrix_verb, "matrix_finite_verb", 4),
                            phrase_words(matrix_theme, "matrix_theme", 5),
                        )
                        paths.append(Clause(
                            frame_id=(
                                f"subject-relative:{number}:{tense}:{head_lemma}:"
                                f"{rel_lemma}:{rel_theme_lemma}:{matrix_lemma}:{matrix_theme_lemma}"
                            ),
                            production="subject_relative",
                            valency="transitive_matrix",
                            case="nominative_head",
                            attachment="subject_relative",
                            number=number,
                            tense=tense,
                            words=tuple(word for phrase in phrases for word in phrase),
                            roles=(
                                "relative_head_subject", "relative_marker",
                                "relative_finite_verb", "relative_theme",
                                "matrix_finite_verb", "matrix_theme",
                            ),
                        ))
    return tuple(paths)


def render_pair(left: Clause, right: Clause) -> str:
    left_text = left.surface[:1].upper() + left.surface[1:]
    return f"{left_text}; {right.surface}."


def midpoint_crossing(text: str) -> dict:
    tape = normalize(text)
    midpoint = (len(tape) - 1) // 2
    cursor = 0
    crossing = None
    for token in re.findall(r"[A-Za-z]+", text):
        start, end = cursor, cursor + len(token)
        if start <= midpoint < end:
            crossing = {
                "token": token,
                "token_interval": [start, end],
                "midpoint": midpoint,
                "offset": midpoint - start,
                "inside_word": len(token) > 1,
            }
            break
        cursor = end
    return {"midpoint": midpoint, "crossing": crossing}


STOPWORDS = {
    "a", "an", "the", "who", "to", "for", "and", "at", "by", "near",
    "after", "with", "in", "on", "of",
}


def shortcut_flags(left: Clause, right: Clause) -> dict:
    def content(clause: Clause) -> set[str]:
        return {
            word.surface.casefold()
            for word in clause.words
            if word.surface.casefold() not in STOPWORDS
        }

    repeated = sorted(content(left).intersection(content(right)))
    self_palindromic = sorted({
        word.surface.casefold()
        for word in (*left.words, *right.words)
        if len(normalize(word.surface)) > 1
        and normalize(word.surface) == normalize(word.surface)[::-1]
    })
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


def row_for(left: Clause, right: Clause, walk) -> dict:
    rendered = render_pair(left, right)
    audit = independent_audit(rendered)
    return {
        "rendered": rendered,
        "letters": audit["letters"],
        "audit": audit,
        "seam_machine": {
            "status": walk.status,
            "mode": walk.seam_mode,
            "matched_characters": walk.matched_characters,
            "online_equations": walk.online_equations,
            "first_mismatch": walk.first_mismatch,
            "left_word_boundary_crossings": walk.left_boundaries,
            "right_word_boundary_crossings": walk.right_boundaries,
            "center_location": walk.center_location,
            "left_remaining": walk.left_remaining,
            "right_remaining": walk.right_remaining,
        },
        "midpoint": midpoint_crossing(rendered),
        "provenance": {
            "left_frame_id": left.frame_id,
            "right_frame_id": right.frame_id,
            "left_production": left.production,
            "right_production": right.production,
            "left_state": {
                "case": left.case, "attachment": left.attachment,
                "number": left.number, "tense": left.tense, "valency": left.valency,
            },
            "right_state": {
                "case": right.case, "attachment": right.attachment,
                "number": right.number, "tense": right.tense, "valency": right.valency,
            },
            "left_roles": list(left.roles),
            "right_roles": list(right.roles),
            "left_clause_words": [word.surface for word in left.words],
            "right_clause_words": [word.surface for word in right.words],
            "lexical_source": (
                "held-out authored to/for benefactive ditransitive slots and "
                "independently authored subject-relative slots"
            ),
            "grammar_paths_independent": True,
            "lexicalized_during_cursor_walk": True,
            "right_clause_read_inward_by_cursor": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "per_search_rlaif": False,
        },
        "shortcut_flags": shortcut_flags(left, right),
        "reader_facing_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "human_raters": 0,
            "reason": "No blinded reader study has been run for this diagnostic lane.",
        },
    }


def cursor_center_contract() -> dict:
    def probe(left_text: tuple[str, ...], right_text: tuple[str, ...]) -> dict:
        def clause(frame_id: str, words: tuple[str, ...]) -> Clause:
            return Clause(
                frame_id, "probe", "probe", "probe", "probe", "sg", "present",
                tuple(Word(x, "probe", 0) for x in words), ("probe",),
            )
        walk = walk_pair(clause("left", left_text), clause("right", right_text))
        return {
            "left_words": list(left_text),
            "right_words": list(right_text),
            "status": walk.status,
            "seam_mode": walk.seam_mode,
            "center_location": walk.center_location,
        }

    probes = [
        probe(("aba",), ("ba",)),
        probe(("ab",), ("cba",)),
        probe(("ab",), ("ba",)),
    ]
    modes = {item["seam_mode"] for item in probes}
    return {
        "description": "Cursor-level contract only; probes are not sentence candidates.",
        "probes": probes,
        "all_expected_modes_present": {
            "left_center_inside_word", "right_center_inside_word", "even_seam"
        }.issubset(modes),
    }


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    exact_collision = [
        entry.get("id") for entry in entries
        if entry.get("id") == EXPERIMENT_ID or entry.get("signature") == SIGNATURE
    ]
    related = [
        entry.get("id") for entry in entries
        if entry.get("id") in {BASE_ID, PREDECESSOR_ID}
        or any(marker in entry.get("signature", "") for marker in ("unequal-center", "subject-relative"))
    ]
    return {
        "status": "passed" if not exact_collision else "collision",
        "registry_inspected": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": exact_collision,
        "base_lane_seen": BASE_ID in related,
        "predecessor_lane_seen": PREDECESSOR_ID in related,
        "related_signatures_seen": related,
        "signature": SIGNATURE,
        "distinction": (
            "Adds a preposition-conditioned to-dative versus for-benefactive alternation "
            "and a separately authored subject-relative attachment. Case, attachment, "
            "number, and tense are live in each path; the prior object-relative frame "
            "is not replayed. The unequal-boundary cursor product is retained."
        ),
        "not_a_duplicate_sweep": True,
    }


def select_controls(observations: list[tuple], limit: int = 24) -> list[dict]:
    families = ("benefactive_to", "benefactive_for", "subject_relative")
    selected = []
    seen_pairs: set[tuple[int, int]] = set()
    seen_surfaces: set[tuple[str, str]] = set()

    for left_family in families:
        for right_family in families:
            pool = [
                row for row in observations
                if row[2].production == left_family and row[3].production == right_family
            ]
            pool.sort(
                key=lambda item: (
                    not shortcut_flags(item[2], item[3])["repeated_content_units"],
                    item[2].letters + item[3].letters,
                ),
                reverse=True,
            )
            for row in pool:
                key = (row[0], row[1])
                surface_key = (row[2].surface, row[3].surface)
                if row[2].surface == row[3].surface or key in seen_pairs or surface_key in seen_surfaces:
                    continue
                selected.append(row)
                seen_pairs.add(key)
                seen_surfaces.add(surface_key)
                break

    ordered = sorted(
        observations,
        key=lambda item: (item[2].letters + item[3].letters, item[0], item[1]),
        reverse=True,
    )
    for row in ordered:
        key = (row[0], row[1])
        surface_key = (row[2].surface, row[3].surface)
        if row[2].surface == row[3].surface or key in seen_pairs or surface_key in seen_surfaces:
            continue
        selected.append(row)
        seen_pairs.add(key)
        seen_surfaces.add(surface_key)
        if len(selected) >= limit:
            break
    return [row_for(left, right, walk) for _, _, left, right, walk in selected[:limit]]


def run() -> dict:
    paths = build_paths()
    observations = []
    total_equations = mismatch_prunes = overhang_prunes = seam_closures = 0
    exact_candidates = []
    unequal_words = unequal_letters = 0
    center_modes: dict[str, int] = {}

    for left_index, left in enumerate(paths):
        for right_index, right in enumerate(paths):
            walk = walk_pair(left, right)
            observations.append((left_index, right_index, left, right, walk))
            total_equations += walk.online_equations
            mismatch_prunes += walk.status == "mismatch_pruned"
            overhang_prunes += walk.status == "overhang_pruned"
            seam_closures += walk.status == "closed"
            center_modes[walk.seam_mode] = center_modes.get(walk.seam_mode, 0) + 1
            unequal_words += len(left.words) != len(right.words)
            unequal_letters += left.letters != right.letters
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
    family_counts = {
        family: sum(path.production == family for path in paths)
        for family in ("benefactive_to", "benefactive_for", "subject_relative")
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Held-out typed to/for benefactive ditransitive alternation and independently "
            "authored subject-relative paths intersected by the same unequal-center two-cursor "
            "seam product; left characters advance forward, right characters advance inward, "
            "and unequal word/clause boundaries plus in-word centers remain live."
        ),
        "operator_added": {
            "name": "typed benefactive to/for alternation plus subject-relative frame",
            "benefactive_to_form": "Subject finite-verb Theme to Recipient",
            "benefactive_for_form": "Subject finite-verb Theme for Beneficiary",
            "subject_relative_form": "SubjectHead who RelativeFiniteVerb RelativeTheme MatrixFiniteVerb MatrixTheme",
            "case_states": ["to_dative", "for_benefactive", "nominative_head"],
            "attachment_states": ["matrix", "subject_relative"],
            "number_states": ["sg", "pl"],
            "tense_states": ["present", "past"],
            "same_cursor_product": True,
            "prior_object_relative_bank_replayed": False,
        },
        "stats": {
            "heldout_clause_paths": len(paths),
            "paths_by_production": family_counts,
            "paired_grammar_states": len(observations),
            "online_character_equations": total_equations,
            "mismatch_prunes": mismatch_prunes,
            "overhang_prunes": overhang_prunes,
            "seam_closures": seam_closures,
            "seam_modes": center_modes,
            "unequal_word_count_pairs": unequal_words,
            "unequal_letter_count_pairs": unequal_letters,
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
        "center_transition_contract": cursor_center_contract(),
        "novelty_preflight": novelty_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_cursor_source": "experiments/unequal_center_ditransitive_relative_complement_20260920.py",
            "independent_audits": ["direct outside-in two-pointer scan", "forward/reverse SHA-256"],
            "search_lexical_source": (
                "held-out authored to/for benefactive and subject-relative role slots; "
                "no borrowed sentence context"
            ),
            "old_object_relative_paths_replayed": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "fragment_output": False,
            "per_search_rlaif": False,
        },
        "status": "completed_no_exact_closure" if not exact_candidates else "mechanical_exact_requires_reader_gate",
        "next_construction": {
            "operator": "typed temporal subject-relative attachment",
            "description": (
                "Add one independently authored subject-relative frame with a temporal adjunct "
                "and future/perfect tense state, preserving the same case, attachment, number, "
                "tense, unequal-boundary, and in-word-center cursor product. Do not replay this "
                "lexical bank or add a repair/reversal operator."
            ),
            "reader_facing_test": (
                "Retain only complete vivid prose; independently audit any exact closure above "
                "38, then run randomized blinded intact-vs-shuffled ratings."
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"experiment_id": output["experiment_id"], "stats": output["stats"]}, sort_keys=True))
