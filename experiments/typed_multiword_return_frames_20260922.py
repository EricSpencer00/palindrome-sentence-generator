"""Intersect typed multiword returns with the live plural-``s`` stack.

This is the bounded successor to ``freshness_register_morphology_cycle``.
It does not add another bare lemma pair.  Instead it reads four independently
typed Brown frames for ``y`` and solves the still-open equation

    T(x) + s = s + reverse(T(y))

against the same four-frame grammar *before* any sentence is rendered.  A
successful pair is then pushed outside the productive ``spoon+s/snoop`` cycle
and the existing inspection/carton carrier.  Return frames are emitted only by
the LIFO pop phase.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from experiments.productive_affix_return_stack_20260922 import (  # noqa: E402
    complementary_boundary_mask,
)
from llm_palindrome.admission import (  # noqa: E402
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)


EXPERIMENT_ID = "typed-multiword-return-frames-20260922"
DEFAULT_OUTPUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
RESIDUAL = "s"
WORD = re.compile(r"^[a-z]+$")
FRAME_TYPES = ("V_PRON", "V_DET_N", "V_PARTICLE", "FINITE_SUBJECT_VERB")
ORDINARY_TWO = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or ox so to up us we".split()
)

CARRIER_LEFT = ("no", "trace", "note")
CARRIER_RIGHT = ("set", "one", "carton")
INNER_X = ("spoon",)
INNER_Y = ("snoop",)

# V+particle is a complete imperative only for an independently licensed
# intransitive use.  The list is deliberately closed; this experiment does not
# turn arbitrary Brown ``VB RP`` adjacency into a sentence.
COMPLETE_PARTICLE_FRAMES = frozenset({
    ("back", "off"), ("break", "away"), ("carry", "on"),
    ("come", "back"), ("come", "in"), ("come", "on"),
    ("go", "ahead"), ("go", "away"), ("go", "back"), ("go", "on"),
    ("hold", "on"), ("look", "out"), ("move", "along"),
    ("move", "on"), ("run", "along"), ("sit", "down"),
    ("stand", "aside"), ("stand", "back"), ("stand", "up"),
    ("stay", "away"), ("stay", "on"), ("step", "aside"),
    ("step", "back"), ("stop", "by"), ("turn", "around"),
    ("wake", "up"), ("walk", "on"), ("watch", "out"),
})

# A two-word finite clause is admitted only when the lexical verb has an
# ordinary objectless reading.  Agreement is checked separately from this
# valency screen.
COMPLETE_FINITE_VERBS = frozenset({
    "agrees", "agreed", "appears", "appeared", "arrives", "arrived",
    "begins", "began", "comes", "came", "continues", "continued",
    "dies", "died", "ends", "ended", "exists", "existed", "falls", "fell",
    "goes", "went", "happens", "happened", "laughs", "laughed",
    "leaves", "left", "lives", "lived", "moves", "moved", "nods", "nodded",
    "pauses", "paused", "remains", "remained", "rests", "rested",
    "returns", "returned", "rises", "rose", "runs", "ran", "sits", "sat",
    "sleeps", "slept", "smiles", "smiled", "speaks", "spoke",
    "stands", "stood", "starts", "started", "stays", "stayed",
    "stops", "stopped", "waits", "waited", "walks", "walked",
    "works", "worked", "is", "are", "was", "were",
})

EVENT_DOMAINS = {
    "inspection": frozenset({
        "check", "examine", "inspect", "look", "mark", "note", "notice",
        "observe", "scan", "search", "see", "show", "snoop", "spot",
        "study", "survey", "test", "trace", "watch",
    }),
    "handling": frozenset({
        "carry", "count", "hold", "keep", "move", "pack", "place", "put",
        "save", "seal", "secure", "send", "set", "shift", "ship", "sort",
        "stack", "store", "stow", "take", "turn",
    }),
    "workflow": frozenset({
        "begin", "continue", "end", "go", "hold", "leave", "move", "pause",
        "return", "run", "stand", "start", "stay", "step", "stop", "wait",
        "walk", "work",
    }),
    "reporting": frozenset({
        "answer", "ask", "note", "report", "say", "show", "signal", "speak",
        "state", "tell",
    }),
}

IRREGULAR_LEMMAS = {
    "is": "be", "are": "be", "was": "be", "were": "be",
    "came": "come", "fell": "fall", "left": "leave", "ran": "run",
    "rose": "rise", "sat": "sit", "slept": "sleep", "spoke": "speak",
    "stood": "stand", "went": "go",
}


def _tag(tag: str) -> str:
    return tag.casefold().split("-", 1)[0].split("+", 1)[0].rstrip("*")


def _is_proper(tag: str) -> bool:
    return _tag(tag).startswith("np")


def _is_noun(tag: str) -> bool:
    return _tag(tag).startswith("nn") and not _is_proper(tag)


def _is_det(tag: str) -> bool:
    return _tag(tag) in {"at", "dt", "dti", "dts", "dtx", "abn", "abx"}


def _is_object_pronoun(tag: str) -> bool:
    return _tag(tag) in {"ppo", "ppl", "ppls"}


def _is_base_verb(tag: str) -> bool:
    return _tag(tag) == "vb"


def _is_particle(tag: str) -> bool:
    return _tag(tag) == "rp"


def _subject_number(word: str, tag: str) -> str | None:
    base = _tag(tag)
    if base in {"pps", "nn"}:
        return "singular"
    if base == "nns":
        return "plural"
    if base == "ppss":
        return "singular" if word == "i" else "plural"
    return None


def _finite_number(tag: str) -> str | None:
    base = _tag(tag)
    if base in {"vbz", "bez", "hvz", "doz"}:
        return "singular"
    if base in {"vb", "ber", "hv", "do", "bem"}:
        return "plural"
    if base in {"vbd", "bed", "bedz", "hvd", "dod", "md"}:
        return "neutral"
    return None


def _agreement(subject_word: str, subject_tag: str, verb_tag: str) -> str | None:
    subject = _subject_number(subject_word, subject_tag)
    verb = _finite_number(verb_tag)
    if subject is None or verb is None:
        return None
    if verb == "neutral" or subject == verb:
        return f"{subject}-subject/{verb}-finite"
    return None


def _base_verb(surface: str) -> str:
    if surface in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[surface]
    for suffix in ("ies", "ied", "ing", "ed", "es", "s"):
        if surface.endswith(suffix) and len(surface) > len(suffix) + 2:
            stem = surface[:-len(suffix)]
            if suffix in {"ies", "ied"}:
                return stem + "y"
            return stem
    return surface


def _event_domains(verb: str) -> tuple[str, ...]:
    lemma = _base_verb(verb)
    return tuple(sorted(name for name, words in EVENT_DOMAINS.items() if lemma in words))


@dataclass(frozen=True)
class Token:
    word: str
    tag: str


@dataclass(frozen=True)
class Frame:
    frame_type: str
    words: tuple[str, ...]
    tags: tuple[str, ...]
    source: str
    sentence: int
    start: int
    agreement: str
    completeness: str

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def verb(self) -> str:
        return self.words[1] if self.frame_type == "FINITE_SUBJECT_VERB" else self.words[0]


def iter_brown_sentences(corpus_dir: Path):
    sentence_id = 0
    for path in sorted(corpus_dir.glob("c*")):
        if not path.is_file():
            continue
        for raw_line in path.read_text(errors="ignore").splitlines():
            sentence: list[Token | None] = []
            for field in raw_line.split():
                if "/" not in field:
                    sentence.append(None)
                    continue
                raw_word, raw_tag = field.rsplit("/", 1)
                word = raw_word.casefold()
                if not WORD.fullmatch(word) or _is_proper(raw_tag):
                    sentence.append(None)
                else:
                    sentence.append(Token(word, raw_tag))
            yield path.name, sentence_id, sentence
            sentence_id += 1


def classify_frame(tokens: tuple[Token, ...], source: str, sentence: int, start: int) -> Frame | None:
    words = tuple(token.word for token in tokens)
    tags = tuple(token.tag for token in tokens)
    if len(tokens) == 2 and _is_base_verb(tags[0]) and _is_object_pronoun(tags[1]):
        return Frame("V_PRON", words, tags, source, sentence, start,
                     "imperative-subject=you", "complete transitive imperative")
    if (len(tokens) == 3 and _is_base_verb(tags[0]) and _is_det(tags[1])
            and _is_noun(tags[2])):
        return Frame("V_DET_N", words, tags, source, sentence, start,
                     "imperative-subject=you", "complete transitive imperative")
    if len(tokens) == 2 and _is_base_verb(tags[0]) and _is_particle(tags[1]):
        completeness = (
            "complete licensed particle imperative"
            if words in COMPLETE_PARTICLE_FRAMES else
            "unlicensed objectless particle frame"
        )
        return Frame("V_PARTICLE", words, tags, source, sentence, start,
                     "imperative-subject=you", completeness)
    if len(tokens) == 2:
        agreement = _agreement(words[0], tags[0], tags[1])
        if agreement:
            completeness = (
                "complete finite intransitive clause"
                if words[1] in COMPLETE_FINITE_VERBS else
                "finite agreement passes but objectless valency is unlicensed"
            )
            return Frame("FINITE_SUBJECT_VERB", words, tags, source, sentence, start,
                         agreement, completeness)
    return None


def build_frame_inventory(corpus_dir: Path) -> tuple[dict[str, dict[str, Frame]], dict[str, set[str]], dict]:
    by_type: dict[str, dict[str, Frame]] = {kind: {} for kind in FRAME_TYPES}
    word_tags: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    corpus_digest = hashlib.sha256()
    for path in sorted(corpus_dir.glob("c*")):
        if path.is_file():
            corpus_digest.update(path.name.encode() + b"\0" + path.read_bytes())
    for source, sentence_id, sentence in iter_brown_sentences(corpus_dir):
        counts["sentences"] += 1
        for token in sentence:
            if token:
                word_tags[token.word].add(_tag(token.tag))
                counts["tokens"] += 1
        for start in range(len(sentence)):
            for width in (2, 3):
                part = sentence[start:start + width]
                if len(part) != width or any(token is None for token in part):
                    continue
                frame = classify_frame(tuple(part), source, sentence_id, start)  # type: ignore[arg-type]
                if frame is None:
                    continue
                counts[f"occurrences_{frame.frame_type}"] += 1
                by_type[frame.frame_type].setdefault(frame.tape, frame)
    for kind in FRAME_TYPES:
        counts[f"unique_{kind}"] = len(by_type[kind])
    return by_type, word_tags, {
        **counts, "brown_raw_sha256": corpus_digest.hexdigest(),
        "frame_types": list(FRAME_TYPES),
    }


def _is_complete(frame: Frame) -> bool:
    return frame.completeness.startswith("complete")


def _lemma(word: str, tags: Iterable[str]) -> str:
    bases = {_tag(tag) for tag in tags}
    if bases.intersection({"vbz", "vbd", "vbg", "vbn", "bez", "ber", "bed", "bedz", "hvz", "hvd", "doz", "dod"}):
        return _base_verb(word)
    if "nns" in bases and word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def frame_lemmas(frame: Frame, word_tags: dict[str, set[str]]) -> tuple[str, ...]:
    return tuple(
        _lemma(word, word_tags.get(word, ()))
        for word in frame.words
        if word not in REPEATABLE_FUNCTION_WORDS
    )


def equation_trace(x: Frame, y: Frame) -> dict:
    right_stream = RESIDUAL + y.tape[::-1]
    left_parts = list(x.words) + [RESIDUAL]
    cursor = 0
    trace = []
    for index, part in enumerate(left_parts):
        expected = right_stream[cursor:cursor + len(part)]
        owner = "x_grammar" if index < len(x.words) else "morphology_residual"
        trace.append({
            "grammar_phase": f"x_token_{index + 1}" if owner == "x_grammar" else "residual_close",
            "owner": owner, "cursor_before": cursor, "emitted": part,
            "opposed": expected, "matched": part == expected,
            "cursor_after": cursor + len(part),
        })
        cursor += len(part)
    left = x.tape + RESIDUAL
    return {
        "equation": "T(x) s = s reverse(T(y))", "left": left,
        "right": right_stream, "holds": left == right_stream,
        "owner_residual_cursor_trace": trace, "closed_before_render": left == right_stream,
    }


def _all_frames(by_type: dict[str, dict[str, Frame]]) -> dict[str, list[Frame]]:
    out: dict[str, list[Frame]] = defaultdict(list)
    for frames in by_type.values():
        for frame in frames.values():
            out[frame.tape].append(frame)
    return out


def _longest_lexical_prefix(target: str, vocabulary: set[str]) -> dict:
    by_first: dict[str, list[str]] = defaultdict(list)
    for word in vocabulary:
        if len(word) == 1 and word not in {"a", "i"}:
            continue
        if len(word) == 2 and word not in ORDINARY_TWO:
            continue
        by_first[word[0]].append(word)
    for words in by_first.values():
        words.sort(key=lambda value: (-len(value), value))
    best = (0, ())
    stack = [(0, ())]
    seen = set()
    while stack:
        cursor, words = stack.pop()
        if cursor > best[0] or (cursor == best[0] and len(words) < len(best[1])):
            best = (cursor, words)
        if (cursor, len(words)) in seen or len(words) >= 4:
            continue
        seen.add((cursor, len(words)))
        for word in by_first.get(target[cursor:cursor + 1], ()):
            if target.startswith(word, cursor):
                stack.append((cursor + len(word), words + (word,)))
    closed_lexically = best[0] == len(target)
    return {
        "target_x_tape": target, "cursor": best[0],
        "lexical_prefix_words": list(best[1]), "unresolved_residual": target[best[0]:],
        "owner": "x_grammar_phase" if closed_lexically else "x_lexical_cursor",
        "failure": (
            "lexical segmentation closes, but no segmentation has one of the four declared frame types"
            if closed_lexically else
            "ordinary lexical segmentation stops before the derived x tape closes"
        ),
    }


def solve_pairs(by_type: dict[str, dict[str, Frame]], word_tags: dict[str, set[str]]) -> tuple[list[dict], dict]:
    x_index = _all_frames(by_type)
    vocabulary = set(word_tags)
    pairs = []
    lane_rows = []
    for kind in FRAME_TYPES:
        stats = Counter()
        best_obstruction = None
        best_cursor = -1
        failures = []
        for y in sorted(by_type[kind].values(), key=lambda frame: (frame.words, frame.tags)):
            stats["typed_y_frames"] += 1
            if not _is_complete(y):
                stats["incomplete_y_rejections"] += 1
                continue
            if not y.tape.startswith(RESIDUAL):
                stats["initial_residual_rejections"] += 1
                continue
            right_stream = RESIDUAL + y.tape[::-1]
            if not right_stream.endswith(RESIDUAL):
                raise AssertionError(y)
            target_x = right_stream[:-len(RESIDUAL)]
            stats["equations_solved_online"] += 1
            xs = x_index.get(target_x, ())
            if not xs:
                stats["no_typed_x"] += 1
                obstruction = _longest_lexical_prefix(target_x, vocabulary)
                obstruction.update({"y_frame": asdict(y), "frame_type": kind})
                if obstruction["cursor"] > best_cursor:
                    best_cursor = obstruction["cursor"]
                    best_obstruction = obstruction
                if len(failures) < 8:
                    failures.append(obstruction)
                continue
            stats["typed_x_tape_hits"] += 1
            for x in xs:
                stats["typed_pair_hits"] += 1
                if not _is_complete(x):
                    stats["incomplete_x_rejections"] += 1
                    continue
                trace = equation_trace(x, y)
                if not trace["holds"]:
                    raise AssertionError((x, y))
                x_lemmas = frame_lemmas(x, word_tags)
                y_lemmas = frame_lemmas(y, word_tags)
                pairs.append({
                    "x": x, "y": y, "x_lemmas": x_lemmas, "y_lemmas": y_lemmas,
                    "equation": trace, "y_lane": kind,
                })
                stats["complete_pairs"] += 1
        lane_rows.append({
            "frame_type": kind, "stats": dict(stats),
            "best_obstruction": best_obstruction,
            "obstruction_samples": failures,
        })
    pairs.sort(key=lambda row: (
        -(len(row["x"].tape) + len(row["y"].tape)),
        row["x"].words, row["y"].words,
    ))
    return pairs, {"lanes": lane_rows, "complete_pairs": len(pairs)}


def independent_audit(text: str) -> dict:
    tape = "".join(re.findall(r"[a-z]", text.casefold()))
    mismatch = next((i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape), "two_pointer_exact": mismatch is None and bool(tape),
        "first_mismatch": mismatch, "sha256_forward": forward,
        "sha256_reverse": reverse, "hashes_agree": forward == reverse,
    }


def proper_palindromic_word_spans(text: str) -> list[dict]:
    words = tokenize(text)
    rows = []
    for start in range(len(words)):
        for stop in range(start + 2, len(words) + 1):
            if start == 0 and stop == len(words):
                continue
            tape = "".join(normalize_letters(word) for word in words[start:stop])
            if tape and tape == tape[::-1]:
                rows.append({"start": start, "stop": stop, "words": list(words[start:stop])})
    return rows


def _sentence(frame: Frame) -> str:
    return " ".join(frame.words).capitalize() + "."


def semantic_event_register(x: Frame, y: Frame) -> dict:
    x_domains = _event_domains(x.verb)
    y_domains = _event_domains(y.verb)
    active = {"inspection", "handling", "workflow"}
    x_connected = bool(active.intersection(x_domains))
    y_connected = bool(active.intersection(y_domains))
    return {
        "scene": "inspect anomalous spoons, report the observation, then pack one carton",
        "carrier_domains": ["inspection", "handling"],
        "inner_event": {"subject": "spoons", "verb": "snoop", "domains": ["inspection"]},
        "x_event": {"verb": x.verb, "domains": list(x_domains), "connected": x_connected},
        "y_event": {"verb": y.verb, "domains": list(y_domains), "connected": y_connected},
        "all_events_connected": x_connected and y_connected,
    }


def render_pair(pair: dict, *, include_command_cycle: bool = False) -> dict:
    x: Frame = pair["x"]
    y: Frame = pair["y"]
    if include_command_cycle:
        middle = f"{_sentence(x)} Spot spoons. Snoop. Stop. {_sentence(y)}"
        left_middle = x.words + ("spot", "spoons")
        right_middle = ("snoop", "stop") + y.words
        stack_pushes = [CARRIER_RIGHT, y.words, ("stop",), INNER_Y]
    else:
        middle = f"{_sentence(x)} Spoons snoop. {_sentence(y)}"
        left_middle = x.words + ("spoons",)
        right_middle = ("snoop",) + y.words
        stack_pushes = [CARRIER_RIGHT, y.words, INNER_Y]
    rendered = f"No trace. Note: {middle} Set one carton."
    left_words = CARRIER_LEFT + left_middle
    right_words = right_middle + CARRIER_RIGHT
    mask = complementary_boundary_mask(left_words, right_words)
    spans = proper_palindromic_word_spans(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=45, max_letters=160)
    audit = independent_audit(rendered)
    event = semantic_event_register(x, y)
    reserved = {"trace", "note", "set", "carton", "spoon", "snoop"}
    if include_command_cycle:
        reserved.update({"spot", "stop"})
    lemmas = list(pair["x_lemmas"] + pair["y_lemmas"])
    all_lemmas = sorted(reserved) + lemmas
    freshness = {
        "carrier_and_inner_lemmas": sorted(reserved), "frame_lemmas": lemmas,
        "all_lemmas_distinct": len(all_lemmas) == len(set(all_lemmas)),
    }
    admitted = (
        audit["two_pointer_exact"] and audit["hashes_agree"]
        and audit["letters"] > 44 and all(checks.values()) and mask["passes"]
        and not spans and freshness["all_lemmas_distinct"]
        and event["all_events_connected"]
    )
    stack_trace = []
    for depth, words in enumerate(stack_pushes, 1):
        stack_trace.append({"operation": "push_return_frame", "words": list(words), "stack_depth": depth})
    for depth, words in enumerate(reversed(stack_pushes)):
        stack_trace.append({"operation": "pop_return_frame", "words": list(words),
                            "stack_depth": len(stack_pushes) - depth - 1})
    return {
        "rendered": rendered, "x_frame": asdict(x), "y_frame": asdict(y),
        "y_lane": pair["y_lane"], "equation": pair["equation"],
        "grammar": {
            "x_phase": x.frame_type, "y_return_phase": y.frame_type,
            "x_completeness": x.completeness, "y_completeness": y.completeness,
            "x_agreement": x.agreement, "y_agreement": y.agreement,
            "all_sentences_complete": True,
        },
        "semantic_event_register": event, "lemma_freshness": freshness,
        "return_stack": {"lifo": True, "trace": stack_trace},
        "complementary_token_boundary_mask": mask,
        "proper_span_mask": {"forbidden_spans": spans, "passes": not spans},
        "central_admission": checks, "independent_audit": audit,
        "mechanically_and_semantically_admitted": admitted,
        "provenance": {
            "open_carrier": True, "typed_frames_joined_before_render": True,
            "finished_palindromic_spans_in_inventory": False,
            "bare_lemma_bank_widened": False, "proper_names": False,
        },
    }


def _compact_pair(pair: dict) -> dict:
    return {
        "x_words": list(pair["x"].words), "x_type": pair["x"].frame_type,
        "y_words": list(pair["y"].words), "y_type": pair["y"].frame_type,
        "y_lane": pair["y_lane"], "equation": pair["equation"],
    }


def _rejection_summary(row: dict) -> dict:
    failed_central = sorted(key for key, passed in row["central_admission"].items() if not passed)
    failed = []
    if not row["lemma_freshness"]["all_lemmas_distinct"]:
        failed.append("global_lemma_freshness")
    if not row["semantic_event_register"]["all_events_connected"]:
        failed.append("semantic_event_register")
    if not row["complementary_token_boundary_mask"]["passes"]:
        failed.append("complementary_boundary_mask")
    if not row["proper_span_mask"]["passes"]:
        failed.append("proper_span_mask")
    if failed_central:
        failed.append("central_admission")
    return {
        "rendered": row["rendered"], "x_frame": row["x_frame"],
        "y_frame": row["y_frame"], "y_lane": row["y_lane"],
        "equation": row["equation"], "failed_gates": failed,
        "failed_central_checks": failed_central,
        "lemma_freshness": row["lemma_freshness"],
        "semantic_event_register": row["semantic_event_register"],
        "complementary_token_boundary_mask": row["complementary_token_boundary_mask"],
        "proper_span_mask": row["proper_span_mask"],
        "independent_audit": row["independent_audit"],
    }


def run(corpus_dir: Path) -> dict:
    started = time.monotonic()
    by_type, word_tags, corpus = build_frame_inventory(corpus_dir)
    pairs, search = solve_pairs(by_type, word_tags)
    rendered = []
    rejected = []
    gate_failures = Counter()
    for pair in pairs:
        for include_command_cycle in (False, True):
            row = render_pair(pair, include_command_cycle=include_command_cycle)
            if row["mechanically_and_semantically_admitted"]:
                rendered.append(row)
            else:
                rejected.append(_rejection_summary(row))
                if not row["lemma_freshness"]["all_lemmas_distinct"]:
                    gate_failures["global_lemma_freshness"] += 1
                if not row["semantic_event_register"]["all_events_connected"]:
                    gate_failures["semantic_event_register"] += 1
                if not row["complementary_token_boundary_mask"]["passes"]:
                    gate_failures["complementary_boundary_mask"] += 1
                if not row["proper_span_mask"]["passes"]:
                    gate_failures["proper_span_mask"] += 1
                if not all(row["central_admission"].values()):
                    gate_failures["central_admission"] += 1
    unique = {}
    for row in sorted(rendered, key=lambda value: (
        -value["independent_audit"]["letters"], value["rendered"],
    )):
        unique.setdefault(normalize_letters(row["rendered"]), row)
    survivors = list(unique.values())
    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": "can a typed multiword return extend the successful r=s stack past 44 letters as complete connected prose?",
        "acceptance_gate": {
            "exact": True, "minimum_letters_exclusive": 44,
            "all_sentences_complete": True, "single_connected_event_register": True,
            "global_lemma_freshness": True, "complementary_boundary_mask": True,
            "proper_span_mask": True, "central_admission": True,
        },
        "fixed_conditions": {
            "carrier": {"left": list(CARRIER_LEFT), "right": list(CARRIER_RIGHT)},
            "inner_productive_cycle": {"x": list(INNER_X), "surface": ["spoons"], "y": list(INNER_Y)},
            "residual": RESIDUAL, "frame_types": list(FRAME_TYPES),
            "lifo_returns": True, "finished_palindromic_spans_excluded": True,
            "bare_lemma_bank_widened": False,
        },
        "corpus": corpus, "search": search,
        "equation_pairs": [_compact_pair(pair) for pair in pairs],
        "gate_failures": dict(gate_failures),
        "admission_obstructions": rejected,
        "survivors": survivors,
        "stats": {
            "equation_pairs": len(pairs), "rendered_survivors": len(survivors),
            "max_survivor_letters": max((row["independent_audit"]["letters"] for row in survivors), default=0),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "verdict": (
            "promote typed-frame survivor for independent reading"
            if survivors else
            "abort this bounded operator: no declared typed frame clears equation, syntax, event, freshness, boundary, and central gates"
        ),
        "provenance": {
            "host": os.uname().nodename, "python": sys.version.split()[0],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audit_every_survivor": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.brown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "stats": payload["stats"], "gate_failures": payload["gate_failures"],
        "lane_stats": [
            {"frame_type": lane["frame_type"], "stats": lane["stats"]}
            for lane in payload["search"]["lanes"]
        ],
        "survivor_renderings": [row["rendered"] for row in payload["survivors"]],
    }, indent=2))


if __name__ == "__main__":
    main()
