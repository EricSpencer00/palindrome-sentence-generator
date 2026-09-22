"""Compile attested compounds and verb-object bigrams into the r=s stack.

This experiment repairs one precise grammar frontier in the productive suffix
stack.  The carrier is ``we / sew``.  An outer cycle must put a Brown-attested
verb-object bigram after ``we`` and return a Brown-attested, plural-headed noun
compound before ``sew``.  WordNet must license the left verb for a direct
object.  The productive inner cycle remains ``spoon+s / snoop``.  Thus a
surviving path has three finite/imperative event clauses rather than treating
``span``, ``sleet``, or ``spa`` as free-standing returns.

Every lexical choice is made before assembly.  Character debt, strict LIFO
order, global lemma freshness, the complementary-boundary mask, and the proper
span mask are checked on every exact path.  The script never reverses a
finished candidate and has no repair or bare-lemma fallback lane.
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
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters  # noqa: E402


EXPERIMENT_ID = "attested-compound-event-stack-20260922"
DEFAULT_OUTPUT = ROOT / "artifacts" / EXPERIMENT_ID / "search.json"
WORD = re.compile(r"^[a-z]+$")
RESIDUAL = "s"
TRANSITIVE_FRAMES = frozenset({5, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 30, 31})
ANIMATE_NOUN_LEXNAMES = frozenset({"noun.animal", "noun.group", "noun.person"})


@dataclass(frozen=True)
class Bigram:
    kind: str
    words: tuple[str, str]
    tags: tuple[str, str]
    count: int
    first_source: str
    verb_lemma: str | None = None
    wordnet_frames: tuple[int, ...] = ()
    noun_lemmas: tuple[str, ...] = ()
    noun_lexnames: tuple[tuple[str, ...], ...] = ()

    @property
    def tape(self) -> str:
        return "".join(self.words)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tag(raw: str) -> str:
    return raw.casefold().split("-", 1)[0].split("+", 1)[0].rstrip("*")


def _is_common_noun(tag: str) -> bool:
    return tag.startswith("nn") and not tag.startswith("np")


def _is_plural_noun(tag: str) -> bool:
    return tag.startswith("nns")


def iter_brown_sentences(corpus_dir: Path):
    sentence = 0
    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or path.name in {"README", "CONTENTS", "cats.txt"}:
            continue
        for line_number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            tokens = []
            for field in line.split():
                if "/" not in field:
                    tokens.append(None)
                    continue
                raw_word, raw_tag = field.rsplit("/", 1)
                word, tag = raw_word.casefold(), _tag(raw_tag)
                if not WORD.fullmatch(word) or tag.startswith(("np", "fw")):
                    tokens.append(None)
                else:
                    tokens.append((word, tag))
            yield path.name, line_number, sentence, tuple(tokens)
            sentence += 1


def _wordnet_lines(wordnet_zip: Path, member: str):
    with zipfile.ZipFile(wordnet_zip) as archive:
        with archive.open(member) as handle:
            for raw in handle:
                yield raw.decode("utf-8").rstrip("\n")


def load_exceptions(wordnet_zip: Path, pos: str) -> dict[str, tuple[str, ...]]:
    out = {}
    for line in _wordnet_lines(wordnet_zip, f"wordnet/{pos}.exc"):
        fields = line.split()
        if fields:
            out[fields[0]] = tuple(fields[1:])
    return out


def load_verb_frames(wordnet_zip: Path) -> dict[str, tuple[int, ...]]:
    frames_by_lemma: dict[str, set[int]] = defaultdict(set)
    for line in _wordnet_lines(wordnet_zip, "wordnet/data.verb"):
        if not line or line.startswith("  "):
            continue
        fields = line.split("|", 1)[0].split()
        if len(fields) < 5:
            continue
        word_count = int(fields[3], 16)
        cursor = 4
        lemmas = []
        for _ in range(word_count):
            lemmas.append(fields[cursor].casefold())
            cursor += 2
        pointer_count = int(fields[cursor])
        cursor += 1 + 4 * pointer_count
        if cursor >= len(fields):
            continue
        frame_count = int(fields[cursor])
        cursor += 1
        per_word: dict[int, set[int]] = defaultdict(set)
        for _ in range(frame_count):
            if cursor + 2 >= len(fields) or fields[cursor] != "+":
                break
            frame_number = int(fields[cursor + 1])
            word_number = int(fields[cursor + 2], 16)
            per_word[word_number].add(frame_number)
            cursor += 3
        for index, lemma in enumerate(lemmas, 1):
            frames_by_lemma[lemma].update(per_word.get(0, ()))
            frames_by_lemma[lemma].update(per_word.get(index, ()))
    return {lemma: tuple(sorted(frames)) for lemma, frames in frames_by_lemma.items()}


def load_noun_lexnames(wordnet_zip: Path) -> dict[str, tuple[str, ...]]:
    lexnames = {}
    for line in _wordnet_lines(wordnet_zip, "wordnet/lexnames"):
        fields = line.split()
        if len(fields) >= 2:
            lexnames[int(fields[0])] = fields[1]
    values: dict[str, set[str]] = defaultdict(set)
    for line in _wordnet_lines(wordnet_zip, "wordnet/data.noun"):
        if not line or line.startswith("  "):
            continue
        fields = line.split("|", 1)[0].split()
        if len(fields) < 5:
            continue
        lexname = lexnames.get(int(fields[1]), f"lexfile.{fields[1]}")
        word_count = int(fields[3], 16)
        cursor = 4
        for _ in range(word_count):
            values[fields[cursor].casefold()].add(lexname)
            cursor += 2
    return {lemma: tuple(sorted(names)) for lemma, names in values.items()}


def noun_lemma(word: str, exceptions: dict[str, tuple[str, ...]], noun_lexnames: dict[str, tuple[str, ...]]) -> str:
    for candidate in exceptions.get(word, ()):
        if candidate in noun_lexnames:
            return candidate
    for candidate in (word, word[:-1] if word.endswith("s") else "",
                      word[:-2] if word.endswith("es") else ""):
        if candidate and candidate in noun_lexnames:
            return candidate
    return word


def build_bigram_inventory(corpus_dir: Path, wordnet_zip: Path) -> tuple[dict[str, list[Bigram]], dict[str, list[Bigram]], dict]:
    verb_frames = load_verb_frames(wordnet_zip)
    noun_lexnames = load_noun_lexnames(wordnet_zip)
    noun_exceptions = load_exceptions(wordnet_zip, "noun")
    raw_counts = Counter()
    first: dict[tuple[str, tuple[str, str], tuple[str, str]], str] = {}
    stats = Counter()
    corpus_digest = hashlib.sha256()
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            corpus_digest.update(path.name.encode() + b"\0" + path.read_bytes())
    for source, line, _sentence, tokens in iter_brown_sentences(corpus_dir):
        stats["sentences"] += 1
        for start in range(len(tokens) - 1):
            first_token, second_token = tokens[start:start + 2]
            if first_token is None or second_token is None:
                continue
            words = (first_token[0], second_token[0])
            tags = (first_token[1], second_token[1])
            if _is_common_noun(tags[0]) and _is_common_noun(tags[1]):
                key = ("noun_compound", words, tags)
                raw_counts[key] += 1
                first.setdefault(key, f"{source}:{line}:{start}")
                stats["noun_compound_occurrences"] += 1
            # A bare/base lexical verb plus an adjacent common-noun object is
            # the only imperative lane.  WordNet must independently license a
            # direct-object frame for that lemma.
            if tags[0] == "vb" and _is_common_noun(tags[1]):
                frames = verb_frames.get(words[0], ())
                stats["verb_object_bigrams_seen"] += 1
                if set(frames).intersection(TRANSITIVE_FRAMES):
                    key = ("verb_object", words, tags)
                    raw_counts[key] += 1
                    first.setdefault(key, f"{source}:{line}:{start}")
                    stats["wordnet_valency_licensed_occurrences"] += 1
                else:
                    stats["wordnet_valency_rejections"] += 1
    compounds: dict[str, list[Bigram]] = defaultdict(list)
    objects: dict[str, list[Bigram]] = defaultdict(list)
    for (kind, words, tags), count in raw_counts.items():
        lemmas = tuple(noun_lemma(word, noun_exceptions, noun_lexnames) for word in words)
        noun_names = tuple(noun_lexnames.get(lemma, ()) for lemma in lemmas)
        row = Bigram(
            kind=kind, words=words, tags=tags, count=count,
            first_source=first[(kind, words, tags)],
            verb_lemma=words[0] if kind == "verb_object" else None,
            wordnet_frames=verb_frames.get(words[0], ()) if kind == "verb_object" else (),
            noun_lemmas=lemmas if kind == "noun_compound" else (lemmas[1],),
            noun_lexnames=noun_names if kind == "noun_compound" else (noun_names[1],),
        )
        (compounds if kind == "noun_compound" else objects)[row.tape].append(row)
    for index in (compounds, objects):
        for rows in index.values():
            rows.sort(key=lambda row: (-row.count, row.words, row.tags))
    stats.update({
        "unique_noun_compounds": sum(len(rows) for rows in compounds.values()),
        "unique_wordnet_licensed_verb_objects": sum(len(rows) for rows in objects.values()),
    })
    return compounds, objects, {**stats, "brown_raw_sha256": corpus_digest.hexdigest(),
                                "wordnet_zip_sha256": hashlib.sha256(wordnet_zip.read_bytes()).hexdigest()}


def cycle_equation(left_tape: str, right_tape: str) -> dict:
    lhs = left_tape + RESIDUAL
    rhs = RESIDUAL + right_tape[::-1]
    return {"equation": "T(verb_object) s = s reverse(T(noun_compound))",
            "left": lhs, "right": rhs, "holds": lhs == rhs}


def complementary_boundary_mask(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> dict:
    left_positions, right_positions, cursor = [], [], 0
    for word in left_words:
        cursor += len(word)
        left_positions.append(cursor)
    cursor = 0
    for word in reversed(right_words):
        cursor += len(word)
        right_positions.append(cursor)
    shared = sorted(set(left_positions).intersection(right_positions))
    allowed = [left_positions[-1]] if left_positions and right_positions and left_positions[-1] == right_positions[-1] else []
    forbidden = [position for position in shared if position not in allowed]
    return {"left": left_positions, "right_from_outer_edge": right_positions,
            "shared": shared, "allowed_terminal": allowed,
            "forbidden_internal": forbidden, "passes": not forbidden}


def proper_span_mask(words: tuple[str, ...]) -> dict:
    spans = []
    for start in range(len(words)):
        for stop in range(start + 2, len(words) + 1):
            if start == 0 and stop == len(words):
                continue
            tape = "".join(words[start:stop])
            if tape == tape[::-1]:
                spans.append({"start": start, "stop": stop, "words": words[start:stop], "tape": tape})
    return {"passes": not spans, "forbidden_spans": spans}


def _fresh_lemmas(verb_object: Bigram, compound: Bigram) -> dict:
    lemmas = (verb_object.verb_lemma, verb_object.noun_lemmas[-1], "spoon", "snoop",
              compound.noun_lemmas[0], compound.noun_lemmas[1], "sew")
    return {"content_lemmas": lemmas, "all_distinct": len(lemmas) == len(set(lemmas))}


def semantic_event_register(verb_object: Bigram, compound: Bigram) -> dict:
    object_domains = set(verb_object.noun_lexnames[-1])
    modifier_domains = set(compound.noun_lexnames[0])
    head_domains = set(compound.noun_lexnames[1])
    shared = sorted(object_domains.intersection(modifier_domains | head_domains))
    head_animate = bool(head_domains.intersection(ANIMATE_NOUN_LEXNAMES))
    head_plural = _is_plural_noun(compound.tags[1])
    events = (
        {"clause": 1, "subject": "we", "predicate": verb_object.words[0],
         "theme": verb_object.words[1], "finite": True, "valency": "WordNet direct-object frame"},
        {"clause": 2, "subject": "spoons", "predicate": "snoop", "finite": True,
         "agreement": "plural subject plus base-form present"},
        {"clause": 3, "subject": " ".join(compound.words), "predicate": "sew", "finite": True,
         "agreement": "plural compound head plus base-form present"},
    )
    connected = bool(shared) and head_animate and head_plural
    return {"events": events, "shared_noun_lexnames": shared,
            "returned_compound_head_animate": head_animate,
            "returned_compound_head_plural": head_plural,
            "all_events_finite": head_plural, "all_events_connected": connected,
            "connection_rule": "left object and returned compound share a WordNet noun domain; returned head is agentive"}


def audit_path(verb_object: Bigram, compound: Bigram) -> dict:
    equation = cycle_equation(verb_object.tape, compound.tape)
    if not equation["holds"]:
        raise AssertionError((verb_object, compound))
    left_words = ("we",) + verb_object.words + ("spoons",)
    # LIFO: snoop is popped first, then the compiled compound, then carrier sew.
    right_words = ("snoop",) + compound.words + ("sew",)
    words = left_words + right_words
    rendered = (f"We {verb_object.words[0]} {verb_object.words[1]}. "
                f"Spoons snoop. {compound.words[0].capitalize()} {compound.words[1]} sew.")
    tape = normalize_letters(rendered)
    mask = complementary_boundary_mask(left_words, right_words)
    spans = proper_span_mask(words)
    freshness = _fresh_lemmas(verb_object, compound)
    events = semantic_event_register(verb_object, compound)
    checks = mechanical_admission_checks(rendered, min_letters=45, max_letters=180)
    failed = []
    if len(tape) <= 44:
        failed.append("minimum_letters_exclusive")
    if tape != tape[::-1]:
        failed.append("exact")
    if not freshness["all_distinct"]:
        failed.append("global_lemma_freshness")
    if not events["all_events_connected"]:
        failed.append("semantic_event_register")
    if not mask["passes"]:
        failed.append("complementary_boundary_mask")
    if not spans["passes"]:
        failed.append("proper_span_mask")
    for key, value in checks.items():
        if not value and key not in {"length_band"}:
            failed.append(f"central_admission:{key}")
    return {
        "verb_object": asdict(verb_object), "noun_compound": asdict(compound),
        "rendered": rendered, "left_words": left_words, "right_words": right_words,
        "letters": len(tape), "normalized_tape": tape, "equation": equation,
        "lifo_trace": [
            {"operation": "push_carrier", "return": ("sew",), "depth": 1},
            {"operation": "push_compound_cycle", "return": compound.words, "depth": 2},
            {"operation": "push_productive_inner", "return": ("snoop",), "depth": 3},
            {"operation": "pop", "return": ("snoop",), "depth": 2},
            {"operation": "pop", "return": compound.words, "depth": 1},
            {"operation": "pop", "return": ("sew",), "depth": 0},
        ],
        "global_lemma_freshness": freshness, "semantic_event_register": events,
        "complementary_boundary_mask": mask, "proper_span_mask": spans,
        "central_admission": checks,
        "independent_audit": {"two_pointer_exact": tape == tape[::-1],
                              "sha256_forward": _sha(tape), "sha256_reverse": _sha(tape[::-1])},
        "failed_gates": sorted(set(failed)), "survives": not failed,
    }


def _best_cursor(objects: dict[str, list[Bigram]], compounds: dict[str, list[Bigram]]) -> dict:
    # Prefer the longest exact character cycle before syntax/semantics.  If no
    # cycle exists, report the closest target at the first character mismatch.
    best = None
    for left_tape, rows in objects.items():
        if not left_tape.startswith(RESIDUAL):
            continue
        target = RESIDUAL + left_tape[1:][::-1]
        for right_tape, right_rows in compounds.items():
            cursor = 0
            for a, b in zip(target, right_tape):
                if a != b:
                    break
                cursor += 1
            key = (cursor, min(len(target), len(right_tape)), len(left_tape), rows[0].count + right_rows[0].count)
            if best is None or key > best[0]:
                best = (key, rows[0], right_rows[0], target, cursor)
    if best is None:
        return {"grammar_phase": "verb_object_cycle_entry", "character_cursor": 0,
                "failure": "no WordNet-licensed Brown verb-object tape begins with residual s"}
    _key, left, right, target, cursor = best
    return {"grammar_phase": "noun_compound_return_lookup", "character_cursor": cursor,
            "verb_object": asdict(left), "best_compound": asdict(right),
            "required_compound_tape": target, "observed_compound_tape": right.tape,
            "unmatched_required_suffix": target[cursor:],
            "unmatched_observed_suffix": right.tape[cursor:],
            "failure": "first character cursor at which the best attested compound misses the exact r=s return"}


def run(corpus_dir: Path, wordnet_zip: Path) -> dict:
    started = time.monotonic()
    compounds, objects, inventory = build_bigram_inventory(corpus_dir, wordnet_zip)
    exact_pairs = []
    counts = Counter()
    for left_tape, left_rows in objects.items():
        counts["licensed_verb_object_tapes"] += 1
        if not left_tape.startswith(RESIDUAL):
            counts["residual_prefix_rejections"] += 1
            continue
        counts["residual_supported_verb_object_tapes"] += 1
        right_target = RESIDUAL + left_tape[1:][::-1]
        right_rows = compounds.get(right_target, ())
        if not right_rows:
            counts["compound_tape_misses"] += 1
            continue
        counts["exact_tape_intersections"] += 1
        for left in left_rows:
            for right in right_rows:
                counts["typed_surface_pairs"] += 1
                exact_pairs.append(audit_path(left, right))
    exact_pairs.sort(key=lambda row: (
        not row["survives"], -row["letters"], len(row["failed_gates"]),
        row["rendered"],
    ))
    survivors = [row for row in exact_pairs if row["survives"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": "Can Brown-attested verb-object and noun-compound bigrams repair the r=s depth-3 syntax frontier above 44 letters?",
        "acceptance_gate": {"exact": True, "minimum_letters_exclusive": 44,
                            "complete_finite_or_imperative_clauses": True,
                            "wordnet_direct_object_valency": True,
                            "semantic_event_register": True, "strict_lifo": True,
                            "global_lemma_freshness": True,
                            "complementary_boundary_mask": True, "proper_span_mask": True},
        "fixed_conditions": {"residual": RESIDUAL, "stack_depth": 3,
                             "carrier": {"left": ("we",), "right": ("sew",)},
                             "inner_productive_cycle": {"left_lemma": "spoon", "left_surface": "spoons", "right": "snoop"},
                             "outer_left_type": "Brown-contiguous base-verb plus common-noun object",
                             "outer_right_type": "Brown-contiguous common-noun compound with plural head",
                             "wordnet_transitive_frames": sorted(TRANSITIVE_FRAMES),
                             "bare_lemma_pairs": False, "proper_names": False,
                             "fragments": False, "catalogue_text": False,
                             "finished_palindromic_units": False, "post_hoc_repair": False,
                             "lexical_widening_after_run": False},
        "inventory": inventory, "search_stats": dict(counts),
        "exact_stack_count": len(exact_pairs),
        "independently_audited_exact_stacks": exact_pairs,
        "survivors": survivors,
        "first_compound_or_valency_obstruction": None if survivors else _best_cursor(objects, compounds),
        "verdict": ("promote independently audited survivors" if survivors else
                    "bounded compound/valency intersection closes with no accepted survivor"),
        "provenance": {"host": os.uname().nodename, "python": sys.version.split()[0],
                       "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "brown_source": str(corpus_dir), "wordnet_source": str(wordnet_zip),
                       "elapsed_seconds": round(time.monotonic() - started, 3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", required=True, type=Path)
    parser.add_argument("--wordnet", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.brown, args.wordnet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"inventory": payload["inventory"], "search_stats": payload["search_stats"],
                      "exact_stack_count": payload["exact_stack_count"],
                      "survivors": len(payload["survivors"]),
                      "first_obstruction": payload["first_compound_or_valency_obstruction"],
                      "elapsed_seconds": payload["provenance"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
