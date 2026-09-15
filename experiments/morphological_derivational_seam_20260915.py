"""Lexicalized morphology/derivation seam search.

This experiment gives each side its own small grammar and its own choices of
natural inflected or derived forms.  A left clause is rendered first; the
right grammar must independently segment the reversed character tape.  The
reverse parse is allowed to cross both ordinary word boundaries and
morpheme-bearing words, so a match is not a list of reversed word pairs.

The finite inventory is authored here rather than mined from Brown or a POS
tagger.  Results are evidence about this bounded construction only: a full
exact row still needs blinded intact-prose and shuffled-control readers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


MIN_LETTERS = 39
MAX_LETTERS = 180
MAX_FRONTIER_ROWS = 240
MAX_PROBES = 80
FAMILY_ID = "morphological-derivational-seam"
STATE_SPACE_SIGNATURE = (
    "lexicalized-morphology|independent-inflection-derivation|"
    "cross-boundary-reverse-tape|same-frame-intact-prose-score"
)


@dataclass(frozen=True)
class Form:
    text: str
    family: str
    lemma: str
    category: str
    morphology: str


@dataclass(frozen=True)
class Surface:
    frame: str
    roles: tuple[str, ...]
    forms: tuple[Form, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(form.text for form in self.forms)

    @property
    def tape(self) -> str:
        return normalize_letters(" ".join(self.words))

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(form.family for form in self.forms if form.category == "content")


def _forms() -> dict[str, tuple[Form, ...]]:
    """Return a deliberately small, hand-authored morphology inventory."""
    def rows(role: str, specs: Iterable[tuple[str, str, str, str]]) -> tuple[Form, ...]:
        return tuple(Form(text, family, lemma, role, morphology) for text, family, lemma, morphology in specs)

    # Each family contributes independent base, inflectional, and derivational
    # forms.  The parser never reverses a form to manufacture another form.
    return {
        "DET_S": rows("function", (("a", "det-a", "a", "article"), ("an", "det-an", "an", "article"), ("the", "det-the", "the", "article"), ("my", "det-my", "my", "possessive"), ("our", "det-our", "our", "possessive"), ("one", "det-one", "one", "quantifier"))),
        "DET_P": rows("function", (("the", "det-the", "the", "article"), ("my", "det-my", "my", "possessive"), ("our", "det-our", "our", "possessive"), ("some", "det-some", "some", "quantifier"), ("these", "det-these", "these", "demonstrative"))),
        "AUX_S": rows("function", (("is", "be", "be", "present-singular"), ("was", "be", "be", "past-singular"))),
        "AUX_P": rows("function", (("are", "be", "be", "present-plural"), ("were", "be", "be", "past-plural"))),
        "COPULA": rows("function", (("is", "be", "be", "present"), ("was", "be", "be", "past"), ("are", "be", "be", "present-plural"))),
        "ADJ": rows("content", (
            ("active", "act", "act", "derivational-adjective"),
            ("careful", "care", "care", "derivational-adjective"),
            ("creative", "create", "create", "derivational-adjective"),
            ("helpful", "help", "help", "derivational-adjective"),
            ("playful", "play", "play", "derivational-adjective"),
            ("readable", "read", "read", "derivational-adjective"),
        )),
        "ADV": rows("content", (
            ("actively", "act", "act", "derivational-adverb"),
            ("carefully", "care", "care", "derivational-adverb"),
            ("creatively", "create", "create", "derivational-adverb"),
            ("helpfully", "help", "help", "derivational-adverb"),
            ("playfully", "play", "play", "derivational-adverb"),
        )),
        "NOUN_S": rows("content", (
            ("actor", "act", "act", "agentive-noun"),
            ("artist", "art", "art", "base-noun"),
            ("creator", "create", "create", "agentive-noun"),
            ("era", "era", "era", "base-noun"),
            ("helper", "help", "help", "agentive-noun"),
            ("painter", "paint", "paint", "agentive-noun"),
            ("reader", "read", "read", "agentive-noun"),
            ("teacher", "teach", "teach", "agentive-noun"),
            ("writer", "write", "write", "agentive-noun"),
        )),
        "NOUN_P": rows("content", (
            ("actors", "act", "act", "plural"),
            ("artists", "art", "art", "plural"),
            ("creators", "create", "create", "plural"),
            ("helpers", "help", "help", "plural"),
            ("painters", "paint", "paint", "plural"),
            ("readers", "read", "read", "plural"),
            ("teachers", "teach", "teach", "plural"),
            ("writers", "write", "write", "plural"),
        )),
        "OBJECT_S": rows("content", (
            ("arena", "arena", "arena", "base-noun"),
            ("canvas", "canvas", "canvas", "base-noun"),
            ("lesson", "lesson", "lesson", "base-noun"),
            ("letter", "letter", "letter", "base-noun"),
            ("mural", "mural", "mural", "base-noun"),
            ("novel", "novel", "novel", "base-noun"),
            ("plan", "plan", "plan", "base-noun"),
            ("report", "report", "report", "base-noun"),
            ("story", "story", "story", "base-noun"),
        )),
        "V3": rows("content", (
            ("acts", "act", "act", "third-person-singular"),
            ("cares", "care", "care", "third-person-singular"),
            ("creates", "create", "create", "third-person-singular"),
            ("helps", "help", "help", "third-person-singular"),
            ("paints", "paint", "paint", "third-person-singular"),
            ("plays", "play", "play", "third-person-singular"),
            ("reads", "read", "read", "third-person-singular"),
            ("writes", "write", "write", "third-person-singular"),
        )),
        "VPAST": rows("content", (
            ("acted", "act", "act", "past"),
            ("cared", "care", "care", "past"),
            ("created", "create", "create", "past"),
            ("helped", "help", "help", "past"),
            ("painted", "paint", "paint", "past"),
            ("played", "play", "play", "past"),
            ("read", "read", "read", "past"),
            ("wrote", "write", "write", "past"),
        )),
        "VPROG": rows("content", (
            ("acting", "act", "act", "progressive"),
            ("caring", "care", "care", "progressive"),
            ("creating", "create", "create", "progressive"),
            ("helping", "help", "help", "progressive"),
            ("painting", "paint", "paint", "progressive"),
            ("playing", "play", "play", "progressive"),
            ("reading", "read", "read", "progressive"),
            ("writing", "write", "write", "progressive"),
        )),
    }


LEXICON = _forms()

# The sides use different role orders.  In particular, the derivational
# adjective/adverb routes are not a reversed copy of the inflectional route.
LEFT_FRAMES: dict[str, tuple[str, ...]] = {
    "agent_past_object": ("DET_S", "ADJ", "NOUN_S", "VPAST", "DET_S", "OBJECT_S"),
    "agent_present_object": ("DET_S", "NOUN_S", "V3", "DET_S", "OBJECT_S"),
    "plural_progressive": ("DET_P", "NOUN_P", "AUX_P", "VPROG", "DET_S", "OBJECT_S"),
    "derived_copular": ("DET_S", "NOUN_S", "COPULA", "ADJ"),
}
RIGHT_FRAMES: dict[str, tuple[str, ...]] = {
    "plain_past_object": ("DET_S", "NOUN_S", "VPAST", "DET_S", "OBJECT_S"),
    "adverbial_agent": ("DET_S", "NOUN_S", "V3", "ADV"),
    "derived_subject": ("DET_S", "ADJ", "NOUN_S", "COPULA", "ADJ"),
    "plural_past_object": ("DET_P", "NOUN_P", "VPAST", "DET_S", "OBJECT_S"),
}


def _valid_surface(forms: tuple[Form, ...], roles: tuple[str, ...]) -> bool:
    if len(forms) != len(roles):
        return False
    for i, role in enumerate(roles):
        if forms[i].category == "content" and forms[i].family in {f.family for f in forms[:i] if f.category == "content"}:
            return False
        if role == "DET_S" and i + 1 < len(roles) and roles[i + 1] in {"ADJ", "NOUN_S", "OBJECT_S"}:
            agreement_index = i + 2 if roles[i + 1] == "ADJ" and i + 2 < len(roles) else i + 1
            if forms[agreement_index].morphology == "plural":
                return False
            next_initial = forms[i + 1].text[:1].lower()
            if roles[i + 1] == "ADJ":
                next_initial = forms[i + 1].text[:1].lower()
            if forms[i].text == "a" and next_initial in "aeiou":
                return False
            if forms[i].text == "an" and next_initial not in "aeiou":
                return False
        if role == "DET_P" and i + 1 < len(roles) and roles[i + 1] == "NOUN_P":
            if forms[i + 1].morphology != "plural":
                return False
        if role in {"AUX_P"} and i + 1 < len(roles) and roles[i + 1] != "VPROG":
            return False
        if role == "COPULA" and i + 1 < len(roles) and roles[i + 1] != "ADJ":
            return False
    return True


def _compile(frame_map: dict[str, tuple[str, ...]]) -> dict[str, list[Surface]]:
    banks: dict[str, list[Surface]] = {}
    for frame, roles in frame_map.items():
        rows: list[Surface] = []

        def walk(index: int, chosen: tuple[Form, ...]) -> None:
            if index == len(roles):
                if _valid_surface(chosen, roles):
                    rows.append(Surface(frame, roles, chosen))
                return
            for form in LEXICON[roles[index]]:
                walk(index + 1, chosen + (form,))

        walk(0, ())
        banks[frame] = rows
    return banks


def _iter_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_strings(child)


def _repository_fingerprint(output: Path | None) -> tuple[set[str], dict[str, Any]]:
    """Fingerprint existing normalized strings, excluding this output path."""
    keys: set[str] = set()
    files = 0
    malformed = 0
    excluded = output.resolve() if output else None
    for base in (ROOT / "runs", ROOT / "data", ROOT / "experiments"):
        for path in sorted(base.rglob("*.json")):
            if excluded and path.resolve() == excluded:
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                malformed += 1
                continue
            files += 1
            for value in _iter_json_strings(payload):
                try:
                    tape = normalize_letters(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= len(tape) <= MAX_LETTERS and tape == tape[::-1]:
                    keys.add(tape)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    return keys, {"json_files_scanned": files, "malformed_json_files_skipped": malformed, "palindrome_tapes": len(keys), "fingerprint_sha256": digest, "output_excluded": bool(output)}


def _two_pointer(tape: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    return {"exact": bool(tape) and not mismatches, "pairs_checked": len(tape) // 2, "mismatches": mismatches[:12]}


def _prose_score(words: tuple[str, ...]) -> dict[str, Any]:
    """Score each independently rendered side; this is diagnostic, not proof."""
    if not words:
        return {"score": 0.0, "mean_word_zipf": 0.0, "mean_join_zipf": 0.0, "phrase_zipf": 0.0}
    word_freqs = [zipf_frequency(word, "en") for word in words]
    joins = [zipf_frequency(f"{a} {b}", "en") for a, b in zip(words, words[1:])]
    mean_word = sum(word_freqs) / len(word_freqs)
    mean_join = sum(joins) / len(joins) if joins else mean_word
    phrase = zipf_frequency(" ".join(words), "en")
    score = 0.50 * mean_word + 0.30 * mean_join + 0.20 * phrase
    return {"score": round(score, 4), "mean_word_zipf": round(mean_word, 4), "mean_join_zipf": round(mean_join, 4), "phrase_zipf": round(phrase, 4), "blinded_reader_required": True}


def _render(left: Surface, right_words: tuple[str, ...]) -> str:
    return " ".join(left.words).capitalize() + "; " + " ".join(right_words) + "."


def _audit(text: str, left: Surface, right: Surface | None, right_words: tuple[str, ...], *, collision: bool = False) -> dict[str, Any]:
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    pointer = _two_pointer(tape)
    left_score = _prose_score(left.words)
    right_score = _prose_score(right_words)
    families = list(left.families) + [f.family for f in (right.forms if right else ()) if f.category == "content"]
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_two_pointer": pointer,
        "central_admission": checks,
        "intact_prose_score": {"left": left_score, "right": right_score, "combined": round((left_score["score"] + right_score["score"]) / 2, 4)},
        "left_frame": left.frame,
        "left_roles": list(left.roles),
        "left_morphology": [{"word": f.text, "lemma": f.lemma, "family": f.family, "morphology": f.morphology} for f in left.forms],
        "right_frame": right.frame if right else None,
        "right_roles": list(right.roles) if right else [],
        "right_morphology": [{"word": f.text, "lemma": f.lemma, "family": f.family, "morphology": f.morphology} for f in (right.forms if right else ())],
        "content_family_disjoint": len(families) == len(set(families)),
        "existing_repository_tape_collision": collision,
        "mechanically_admitted": pointer["exact"] and all(checks.values()) and not collision and len(families) == len(set(families)),
        "reader_status": "not_run; intact-prose score is diagnostic and not human evidence",
    }


def _partial_parse(tape: str, roles: tuple[str, ...], cap: int = 12) -> tuple[dict[str, Any], ...]:
    states: list[tuple[int, int, tuple[Form, ...]]] = [(0, 0, ())]
    all_states: list[tuple[int, int, tuple[Form, ...]]] = list(states)
    for index, role in enumerate(roles):
        next_states: list[tuple[int, int, tuple[Form, ...]]] = []
        for offset, _, chosen in states:
            for form in LEXICON[role]:
                if tape.startswith(form.text, offset):
                    next_states.append((offset + len(form.text), index + 1, chosen + (form,)))
        states = next_states
        all_states.extend(states)
        if not states:
            break
    all_states.sort(key=lambda row: (-row[0], len(row[2]), tuple(form.text for form in row[2])))
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for offset, index, chosen in all_states:
        key = (offset, tuple(form.text for form in chosen))
        if key in seen:
            continue
        seen.add(key)
        out.append({"consumed": offset, "next_role": roles[index] if index < len(roles) else None, "forms": chosen})
        if len(out) >= cap:
            break
    return tuple(out)


def _seam_diagnostic(left: Surface, parse: dict[str, Any]) -> dict[str, Any]:
    reverse_words = tuple(reversed(left.words))
    source_boundaries = set()
    cursor = 0
    for word in reverse_words:
        cursor += len(normalize_letters(word))
        source_boundaries.add(cursor)
    cursor = 0
    right_boundaries = []
    for form in parse["forms"]:
        cursor += len(form.text)
        right_boundaries.append(cursor)
    crossing = [boundary for boundary in right_boundaries if boundary not in source_boundaries]
    return {"reverse_prefix_letters": parse["consumed"], "right_word_boundaries": right_boundaries, "left_reverse_boundaries": sorted(source_boundaries), "crosses_left_word_boundary": bool(crossing), "crossing_boundaries": crossing, "matched_morphology": [{"word": f.text, "lemma": f.lemma, "morphology": f.morphology} for f in parse["forms"]]}


def _repair_operator(left: Surface, parse: dict[str, Any], right_frame: str) -> dict[str, Any]:
    target_role = parse["next_role"] or (RIGHT_FRAMES[right_frame][-1] if RIGHT_FRAMES[right_frame] else "OBJECT_S")
    matched = parse["forms"]
    seed = matched[-1] if matched else None
    sibling_forms = [form.text for form in LEXICON[target_role] if seed and form.family == seed.family and form.text != seed.text]
    if not sibling_forms:
        sibling_forms = [form.text for form in LEXICON[target_role][:3]]
    return {
        "operator": "same-family-inflection-derivation-seam-repair",
        "action": "replace exactly one right-side form at the first reverse-unmatched character, retain its frame and agreement, then re-run full reverse segmentation and admission",
        "right_frame": right_frame,
        "target_role": target_role,
        "first_reverse_unmatched_offset": parse["consumed"],
        "sibling_forms_to_try": sibling_forms,
        "preserve": ["independent right-side lexical choice", "number/auxiliary agreement", "cross-boundary seam audit", "intact-prose score"],
    }


def run(output: Path | None = None) -> dict[str, Any]:
    existing, fingerprint = _repository_fingerprint(output)
    left_banks = _compile(LEFT_FRAMES)
    right_banks = _compile(RIGHT_FRAMES)
    right_index: dict[str, list[Surface]] = defaultdict(list)
    for rows in right_banks.values():
        for surface in rows:
            right_index[surface.tape].append(surface)

    stats = Counter()
    exact_rows: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame_name, rows in left_banks.items():
        stats["left_surfaces"] += len(rows)
        # Every exact lookup is exhaustive over this finite inventory.  The
        # partial seam frontier is deliberately bounded only for diagnostics.
        for left in rows:
            matches = right_index.get(left.tape[::-1], ())
            if matches:
                stats["reverse_exact_surface_matches"] += len(matches)
            else:
                stats["reverse_exact_surface_misses"] += 1
            for right in matches:
                rendered = _render(left, right.words)
                tape = normalize_letters(rendered)
                if tape in seen:
                    continue
                seen.add(tape)
                collision = tape in existing
                audit = _audit(rendered, left, right, right.words, collision=collision)
                audit["right_independent_choice"] = True
                audit["right_frame_source"] = right.frame
                exact_rows.append(audit)
                stats["exact_rows"] += 1
                if audit["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1

        # Retain a deterministic seam frontier for the first left rows in
        # each frame; this records where the concrete one-form repair acts.
        for left in rows[:MAX_FRONTIER_ROWS]:
            reverse_tape = left.tape[::-1]
            for right_frame, right_roles in RIGHT_FRAMES.items():
                parses = _partial_parse(reverse_tape, right_roles)
                best = parses[0]
                if best["consumed"] == len(reverse_tape) and best["next_role"] is None:
                    continue
                stats["partial_frontier_attempts"] += 1
                text = _render(left, tuple(form.text for form in best["forms"]))
                probe = _audit(text, left, None, tuple(form.text for form in best["forms"]))
                probe["kind"] = "partial_reverse_seam_probe"
                probe["right_frame"] = right_frame
                probe["reverse_tape_first_unmatched"] = reverse_tape[best["consumed"]:best["consumed"] + 16]
                probe["seam_diagnostic"] = _seam_diagnostic(left, best)
                probe["repair_operator"] = _repair_operator(left, best, right_frame)
                frontier.append(probe)

    exact_rows.sort(key=lambda row: (-row["letters"], -row["intact_prose_score"]["combined"], row["rendered"]))
    frontier.sort(key=lambda row: (-row["seam_diagnostic"]["reverse_prefix_letters"], row["rendered"]))
    probe_seen: set[str] = set()
    for row in frontier:
        if row["rendered"] in probe_seen:
            continue
        probe_seen.add(row["rendered"])
        probes.append({
            "rendered": row["rendered"],
            "right_frame": row["right_frame"],
            "reverse_prefix_letters": row["seam_diagnostic"]["reverse_prefix_letters"],
            "crosses_left_word_boundary": row["seam_diagnostic"]["crosses_left_word_boundary"],
            "independent_two_pointer": row["independent_two_pointer"],
            "central_admission": row["central_admission"],
            "intact_prose_score": row["intact_prose_score"],
            "repair_operator": row["repair_operator"],
        })
        if len(probes) >= MAX_PROBES:
            break

    return {
        "status": "no_closure_pivot_required" if not exact_rows else "exact_closures_need_blinded_readers",
        "family_id": FAMILY_ID,
        "state_space_signature": STATE_SPACE_SIGNATURE,
        "config": {
            "construction": "independent lexicalized inflection/derivation clauses joined by a reverse character seam",
            "left_frames": {name: list(roles) for name, roles in LEFT_FRAMES.items()},
            "right_frames": {name: list(roles) for name, roles in RIGHT_FRAMES.items()},
            "inventory_counts": {role: len(forms) for role, forms in LEXICON.items()},
            "morphology_is_hand_authored": True,
            "brown_or_pos_source_used": False,
            "cross_word_and_morpheme_seams": True,
            "independent_right_resegmentation": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "frontier_limit_per_left_frame": MAX_FRONTIER_ROWS,
            "preexisting_tape_scope": "palindromic normalized strings in runs/**/*.json, data/**/*.json, experiments/**/*.json",
            "output_excluded_before_fingerprint": True,
        },
        "repository_fingerprint": fingerprint,
        "novelty_audit": {
            "existing_palindrome_tapes": len(existing),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "output_excluded": fingerprint["output_excluded"],
            "exact_rows_checked_against_existing_tapes": True,
            "admitted_rows_novel": all(row["existing_repository_tape_collision"] is False for row in exact_rows if row["mechanically_admitted"]),
        },
        "stats": dict(stats),
        "exact_closures": exact_rows,
        "admitted": [row for row in exact_rows if row["mechanically_admitted"]],
        "rendered_probes": probes,
        "residual_frontier": frontier[:MAX_PROBES],
        "next_operator": (
            "For the highest-scoring seam probe, replace exactly one right-side inflectional or derivational form with a same-family sibling in the recorded target role; retain agreement, re-segment the full reverse tape, and rerun the two-pointer and admission checks."
            if not exact_rows else
            "Send exact closures to blinded intact-prose and shuffled-control readers; the score and mechanical gate are not readability evidence."
        ),
        "closure_conclusion": (
            "No full reverse parse closed in this finite morphology/derivation grammar. The artifact preserves the longest independent seam probes and their first repair target; the zero is not a claim about the whole English lexicon."
            if not exact_rows else "Exact closures exist but remain unreviewed."
        ),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inventory": "hand-authored lexical families with inflectional and derivational labels",
            "source_text_copied": False,
            "brown_used": False,
            "pos_tagging_used": False,
            "catalogue_reuse": False,
            "output_excluded_from_fingerprint": True,
            "readability_certificate": False,
        },
        "reader_gate": "No programmatic output is human evidence; any mechanically admitted closure requires randomized blinded intact-prose and matched word-shuffle controls.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "stats": result["stats"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted"]), "probes": len(result["rendered_probes"])}, indent=2))


if __name__ == "__main__":
    main()
