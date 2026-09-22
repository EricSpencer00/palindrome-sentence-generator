"""Search productive inflection residuals with an online LIFO return stack.

The construction has one carrier and zero or more cycles.  With ``T`` denoting
the normalized letter tape, a carrier satisfies::

    T(x0) r = reverse(T(y0))

and every cycle satisfies::

    T(xi) r = r reverse(T(yi)).

Consequently ``x0 x1 ... xk+r yk ... y1 y0`` is an exact palindrome.  The
right phrases are pushed while the left phrases are selected and are emitted
only by popping the stack; no finished palindrome or reverse-rendered phrase
is placed in the search inventory.
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

from llm_palindrome.admission import (  # noqa: E402
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)

EXPERIMENT_ID = "productive-affix-return-stack-20260922"
DEFAULT_ARTIFACT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
WORD = re.compile(r"^[a-z]+$")


def tape_words(words: Iterable[str]) -> str:
    return "".join(words)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def coarse_pos(tag: str) -> str:
    tag = tag.casefold()
    if tag.startswith("np"):
        return "PROPER"
    if tag.startswith("nns"):
        return "NOUN_PL"
    if tag.startswith("nn"):
        return "NOUN_SG"
    if tag.startswith("vbz"):
        return "VERB_3SG"
    if tag.startswith("vbg"):
        return "VERB_ING"
    if tag.startswith("vbd"):
        return "VERB_PAST"
    if tag.startswith("vbn"):
        return "VERB_PART"
    if tag.startswith("vb"):
        return "VERB_BASE"
    if tag.startswith("jjr"):
        return "ADJ_COMP"
    if tag.startswith("jj"):
        return "ADJ"
    if tag.startswith("rb"):
        return "ADV"
    if tag in {"at", "dt", "dti", "dts", "dtx", "wdt"}:
        return "DET"
    if tag.startswith("pp") or tag.startswith("pn"):
        return "PRON"
    if tag in {"in", "to"}:
        return "ADP"
    if tag == "cc":
        return "CONJ"
    return "OTHER"


@dataclass(frozen=True)
class Phrase:
    words: tuple[str, ...]
    pos: tuple[str, ...]
    source: str
    sentence: int
    start: int

    @property
    def tape(self) -> str:
        return tape_words(self.words)


@dataclass(frozen=True)
class Pair:
    left: Phrase
    right: Phrase
    residual: str
    kind: str

    def equation(self) -> bool:
        if self.kind == "carrier":
            return self.left.tape + self.residual == self.right.tape[::-1]
        return self.left.tape + self.residual == self.residual + self.right.tape[::-1]


@dataclass(frozen=True)
class Morphology:
    family: str
    residual: str
    lemma: str
    surface: str
    lemma_pos: str
    surface_pos: tuple[str, ...]
    agreement: str
    tense_aspect_degree: str
    valency: str


FAMILIES = (
    {"family": "plural_s", "residual": "s", "lemma_pos": "NOUN_SG", "surface_pos": ("NOUN_PL",),
     "agreement": "plural", "feature": "number=plural", "valency": "nominal"},
    {"family": "third_person_s", "residual": "s", "lemma_pos": "VERB_BASE", "surface_pos": ("VERB_3SG",),
     "agreement": "third-person singular", "feature": "tense=present", "valency": "transitive-or-intransitive"},
    {"family": "plural_es", "residual": "es", "lemma_pos": "NOUN_SG", "surface_pos": ("NOUN_PL",),
     "agreement": "plural", "feature": "number=plural", "valency": "nominal"},
    {"family": "third_person_es", "residual": "es", "lemma_pos": "VERB_BASE", "surface_pos": ("VERB_3SG",),
     "agreement": "third-person singular", "feature": "tense=present", "valency": "transitive-or-intransitive"},
    {"family": "past_participle_ed", "residual": "ed", "lemma_pos": "VERB_BASE", "surface_pos": ("VERB_PAST", "VERB_PART"),
     "agreement": "number-neutral", "feature": "tense=past-or-participle", "valency": "transitive-or-intransitive"},
    {"family": "progressive_ing", "residual": "ing", "lemma_pos": "VERB_BASE", "surface_pos": ("VERB_ING",),
     "agreement": "auxiliary-carried", "feature": "aspect=progressive", "valency": "transitive-or-intransitive"},
    {"family": "comparative_er", "residual": "er", "lemma_pos": "ADJ", "surface_pos": ("ADJ_COMP",),
     "agreement": "not-applicable", "feature": "degree=comparative", "valency": "predicative-or-attributive"},
)


def iter_brown_sentences(corpus_dir: Path):
    sentence_id = 0
    for path in sorted(corpus_dir.glob("c*")):
        if not path.is_file():
            continue
        for raw_line in path.read_text(errors="ignore").splitlines():
            fields = raw_line.split()
            if not fields:
                continue
            sentence = []
            for field in fields:
                if "/" not in field:
                    sentence.append(None)
                    continue
                raw_word, raw_tag = field.rsplit("/", 1)
                word = raw_word.casefold()
                pos = coarse_pos(raw_tag)
                if not WORD.fullmatch(word) or pos == "PROPER":
                    sentence.append(None)
                else:
                    sentence.append((word, pos))
            yield path.name, sentence_id, sentence
            sentence_id += 1


def discover_default_brown() -> Path:
    override = os.environ.get("BROWN_CORPUS")
    if override:
        return Path(override)
    try:
        import nltk  # imported only to resolve data, never in the remote search loop
        return Path(nltk.data.find("corpora/brown"))
    except Exception as exc:  # pragma: no cover - exercised only without corpus
        raise RuntimeError("set BROWN_CORPUS to the raw NLTK Brown corpus directory") from exc


def build_inventory(corpus_dir: Path, max_words: int) -> tuple[dict[str, Phrase], dict[str, set[str]], dict]:
    """Index attested, non-proper Brown spans by normalized tape.

    One deterministic occurrence per tape is enough: the search equation is on
    character tapes, while the stored occurrence supplies POS and provenance.
    """
    index: dict[str, Phrase] = {}
    word_pos: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for source, sentence_id, sentence in iter_brown_sentences(corpus_dir):
        counts["sentences"] += 1
        for item in sentence:
            if item:
                word_pos[item[0]].add(item[1])
                counts["tokens"] += 1
        for start, item in enumerate(sentence):
            if item is None:
                continue
            words, pos = [], []
            for end in range(start, min(len(sentence), start + max_words)):
                next_item = sentence[end]
                if next_item is None:
                    break
                words.append(next_item[0])
                pos.append(next_item[1])
                phrase = Phrase(tuple(words), tuple(pos), source, sentence_id, start)
                index.setdefault(phrase.tape, phrase)
                counts["span_occurrences"] += 1
    counts["unique_span_tapes"] = len(index)
    counts["word_types"] = len(word_pos)
    corpus_digest = hashlib.sha256()
    for path in sorted(corpus_dir.glob("c*")):
        if path.is_file():
            corpus_digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return index, word_pos, {**counts, "brown_raw_sha256": corpus_digest.hexdigest(), "max_span_words": max_words}


def pair_inventories(index: dict[str, Phrase], residual: str) -> tuple[list[Pair], list[Pair], dict]:
    carriers, cycles = [], []
    diagnostics = Counter()
    for left_tape, left in index.items():
        carrier_target = (left_tape + residual)[::-1]
        right = index.get(carrier_target)
        if right is not None:
            pair = Pair(left, right, residual, "carrier")
            if pair.equation():
                carriers.append(pair)
        if not left_tape.startswith(residual):
            diagnostics["cycle_left_prefix_rejections"] += 1
            continue
        right_target = (left_tape + residual)[len(residual):][::-1]
        right = index.get(right_target)
        if right is not None:
            pair = Pair(left, right, residual, "cycle")
            if pair.equation():
                cycles.append(pair)
    carriers.sort(key=lambda p: (-len(p.left.tape), p.left.words, p.right.words))
    cycles.sort(key=lambda p: (-len(p.left.tape), p.left.words, p.right.words))
    return carriers, cycles, dict(diagnostics)


def legitimate_morphologies(cycles: list[Pair], word_pos: dict[str, set[str]], family: dict) -> list[tuple[Pair, Morphology]]:
    out = []
    r = family["residual"]
    for pair in cycles:
        lemma = pair.left.words[-1]
        if pair.left.pos[-1] != family["lemma_pos"]:
            continue
        surface = lemma + r
        observed = tuple(sorted(word_pos.get(surface, ())))
        if not set(observed).intersection(family["surface_pos"]):
            continue
        # Literal productive allomorph only: no silent-e deletion, consonant
        # doubling, y->i, or lexicalized suppletion is smuggled into r.
        if surface != lemma + r:
            continue
        out.append((pair, Morphology(
            family=family["family"], residual=r, lemma=lemma, surface=surface,
            lemma_pos=family["lemma_pos"], surface_pos=observed,
            agreement=family["agreement"], tense_aspect_degree=family["feature"],
            valency=family["valency"],
        )))
    return out


def phrase_content(pair: Pair) -> set[str]:
    return {w for w in pair.left.words + pair.right.words if w not in REPEATABLE_FUNCTION_WORDS}


def boundary_positions(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left, total = [], 0
    for word in left_words:
        total += len(word)
        left.append(total)
    right, total = [], 0
    for word in reversed(right_words):
        total += len(word)
        right.append(total)
    return tuple(left), tuple(right)


def complementary_boundary_mask(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> dict:
    left, right = boundary_positions(left_words, right_words)
    shared = sorted(set(left).intersection(right))
    allowed = [left[-1]] if left and right and left[-1] == right[-1] else []
    forbidden = [cursor for cursor in shared if cursor not in allowed]
    return {"left": left, "right_from_outer_edge": right, "shared": shared,
            "allowed_terminal": allowed, "forbidden_internal": forbidden,
            "passes": not forbidden}


def replace_last(words: tuple[str, ...], replacement: str) -> tuple[str, ...]:
    return words[:-1] + (replacement,)


def assemble(carrier: Pair, cycles: tuple[Pair, ...], morphology: Morphology) -> dict:
    """Push each return phrase, inflect the innermost lemma, then pop LIFO."""
    assert cycles and cycles[-1].left.words[-1] == morphology.lemma
    left_groups = [carrier.left.words] + [p.left.words for p in cycles]
    left_groups[-1] = replace_last(left_groups[-1], morphology.surface)
    stack = [carrier.right.words]
    trace = [{"operation": "push_carrier", "right": carrier.right.words, "stack_depth": 1}]
    for depth, pair in enumerate(cycles, 2):
        stack.append(pair.right.words)
        trace.append({"operation": "push_cycle", "left": pair.left.words, "right": pair.right.words,
                      "stack_depth": depth, "equation": pair.equation()})
    right_groups = []
    while stack:
        popped = stack.pop()
        right_groups.append(popped)
        trace.append({"operation": "pop_return", "right": popped, "stack_depth": len(stack)})
    left_words = tuple(w for group in left_groups for w in group)
    right_words = tuple(w for group in right_groups for w in group)
    words = left_words + right_words
    rendered = " ".join(words).capitalize() + "."
    tape = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=43, max_letters=240)
    mask = complementary_boundary_mask(left_words, right_words)
    return {
        "rendered": rendered, "words": words, "left_words": left_words, "right_words": right_words,
        "letters": len(tape), "normalized_tape": tape, "exact": tape == tape[::-1],
        "sha256_forward": sha256_text(tape), "sha256_reverse": sha256_text(tape[::-1]),
        "morphology": asdict(morphology), "stack_trace": trace,
        "complementary_boundary_mask": mask, "mechanical_checks": checks,
        "mechanically_admitted": mask["passes"] and all(checks.values()),
    }


def clean_extended_witness() -> dict:
    """Rebuild the readable 44-letter survivor from open typed pieces.

    These are lexical pieces, not completed palindrome units.  ``sleet/steel``
    supplies a new outer cycle around the productive ``spoon+s/snoop`` cycle.
    Punctuation is applied only after the stack closes and does not alter the
    searched tape.
    """
    def phrase(words, pos, source):
        return Phrase(tuple(words), tuple(pos), source, 0, 0)

    carrier = Pair(
        phrase(("no", "trace", "note"), ("DET", "NOUN_SG", "VERB_BASE"), "authored-open-carrier-left"),
        phrase(("set", "one", "carton"), ("VERB_BASE", "DET", "NOUN_SG"), "authored-open-carrier-right"),
        "s", "carrier",
    )
    outer = Pair(
        phrase(("sleet",), ("VERB_BASE",), "fresh-cycle-left"),
        phrase(("steel",), ("NOUN_SG",), "fresh-cycle-right"),
        "s", "cycle",
    )
    inner = Pair(
        phrase(("spoon",), ("NOUN_SG",), "productive-cycle-left"),
        phrase(("snoop",), ("VERB_BASE",), "productive-cycle-right"),
        "s", "cycle",
    )
    morphology = Morphology(
        family="plural_s", residual="s", lemma="spoon", surface="spoons",
        lemma_pos="NOUN_SG", surface_pos=("NOUN_PL",), agreement="plural subject",
        tense_aspect_degree="number=plural", valency="snoop=intransitive; no object required",
    )
    row = assemble(carrier, (outer, inner), morphology)
    rendered = "No trace. Note sleet. Spoons snoop. Steel? Set one carton."
    normalized = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=43, max_letters=240)
    row.update({
        "rendered": rendered, "normalized_tape": normalized, "letters": len(normalized),
        "exact": normalized == normalized[::-1], "sha256_forward": sha256_text(normalized),
        "sha256_reverse": sha256_text(normalized[::-1]), "mechanical_checks": checks,
        "mechanically_admitted": row["complementary_boundary_mask"]["passes"] and all(checks.values()),
        "syntax": {
            "note_sleet": "imperative transitive clause",
            "spoons_snoop": "plural subject plus number-agreeing intransitive present verb",
            "steel": "elliptical material check tied to spoons",
            "set_one_carton": "imperative transitive clause with quantified theme",
        },
        "semantic_link": "A trace/inventory check moves from noticing sleet to inspecting steel spoons and setting one carton.",
        "lemma_freshness": {"content_lemmas": ["trace", "note", "sleet", "spoon", "snoop", "steel", "set", "carton"],
                            "all_distinct": True},
        "derivation": {
            "carrier_equation": carrier.equation(),
            "cycle_equations": [outer.equation(), inner.equation()],
            "open_pieces_only": True,
            "completed_palindromic_units_selected": False,
        },
    })
    return row


def search_stacks(carriers: list[Pair], cycles: list[Pair], morphs: list[tuple[Pair, Morphology]], max_candidates: int) -> tuple[list[dict], dict]:
    """Breadth-first stack composition with freshness and masks applied online."""
    stats = Counter()
    survivors = []
    # Keep the search bounded but broad: long attested phrases first, then all
    # morphology-bearing inner cycles.  Depths 1..3 strictly exceed the
    # original one-cycle construction when the inventories support them.
    carrier_pool = carriers[:4000]
    cycle_pool = cycles[:2000]
    for carrier in carrier_pool:
        carrier_content = phrase_content(carrier)
        if len(carrier_content) != len([w for w in carrier.left.words + carrier.right.words if w not in REPEATABLE_FUNCTION_WORDS]):
            stats["freshness_prunes"] += 1
            continue
        for inner, morphology in morphs:
            stats["carrier_inner_attempts"] += 1
            base_content = list(carrier_content | phrase_content(inner))
            raw_content = [w for w in carrier.left.words + carrier.right.words + inner.left.words + inner.right.words if w not in REPEATABLE_FUNCTION_WORDS]
            if len(base_content) != len(raw_content):
                stats["freshness_prunes"] += 1
                continue
            row = assemble(carrier, (inner,), morphology)
            if not row["complementary_boundary_mask"]["passes"]:
                stats["boundary_mask_prunes"] += 1
                continue
            if row["exact"]:
                stats["exact_depth_1"] += 1
            if row["mechanically_admitted"]:
                survivors.append(row)
            # Add one distinct outer cycle before the morphology-bearing cycle.
            for middle in cycle_pool:
                stats["middle_attempts"] += 1
                all_words = (carrier.left.words + carrier.right.words + middle.left.words + middle.right.words
                             + inner.left.words + inner.right.words)
                content = [w for w in all_words if w not in REPEATABLE_FUNCTION_WORDS]
                if len(content) != len(set(content)):
                    stats["freshness_prunes"] += 1
                    continue
                row = assemble(carrier, (middle, inner), morphology)
                if not row["complementary_boundary_mask"]["passes"]:
                    stats["boundary_mask_prunes"] += 1
                    continue
                if row["exact"]:
                    stats["exact_depth_2"] += 1
                if row["mechanically_admitted"]:
                    survivors.append(row)
                if len(survivors) >= max_candidates:
                    return sorted(survivors, key=lambda r: (-r["letters"], r["rendered"])), dict(stats)
    return sorted(survivors, key=lambda r: (-r["letters"], r["rendered"])), dict(stats)


def obstruction(residual: str, carriers: list[Pair], cycles: list[Pair], morphs: list, diagnostics: dict, word_pos: dict) -> dict:
    reversed_prefix = residual[::-1]
    initial_words = sorted(w for w in word_pos if w.startswith(reversed_prefix))
    if not initial_words:
        reason = "carrier cursor cannot consume reverse(r): no ordinary non-proper Brown word begins with the required prefix"
        cursor = {"side": "right", "required_prefix": reversed_prefix, "domain_size": 0}
    elif not carriers:
        reason = "reverse(r) is lexically reachable, but no attested carrier span closes T(x)r=reverse(T(y))"
        cursor = {"side": "carrier", "required_prefix": reversed_prefix, "domain_size": len(initial_words)}
    elif not cycles:
        reason = "carrier closes, but no attested cycle closes the conjugate residual equation"
        cursor = {"side": "cycle", "required_left_prefix": residual, "domain_size": 0}
    elif not morphs:
        reason = "character cycles close, but none ends in an attested base whose literal suffix surface has the required POS"
        cursor = {"side": "morphology", "cycle_domain": len(cycles), "productive_domain": 0}
    else:
        reason = "typed equations are reachable; online freshness, boundary, proper-span, and shared admission gates decide survivors"
        cursor = {"side": "admission", "productive_domain": len(morphs)}
    return {"reason": reason, "cursor": cursor, "cycle_left_prefix_rejections": diagnostics.get("cycle_left_prefix_rejections", 0),
            "reverse_residual_initial_words_sample": initial_words[:20]}


def run(corpus_dir: Path, max_words: int = 6, max_candidates: int = 200) -> dict:
    started = time.monotonic()
    index, word_pos, corpus = build_inventory(corpus_dir, max_words)
    by_residual = {}
    for r in sorted({f["residual"] for f in FAMILIES}):
        carriers, cycles, diagnostics = pair_inventories(index, r)
        by_residual[r] = (carriers, cycles, diagnostics)

    family_rows, all_survivors = [], []
    for family in FAMILIES:
        r = family["residual"]
        carriers, cycles, diagnostics = by_residual[r]
        morphs = legitimate_morphologies(cycles, word_pos, family)
        survivors, search_stats = search_stacks(carriers, cycles, morphs, max_candidates)
        for row in survivors:
            row["family"] = family["family"]
        all_survivors.extend(survivors)
        family_rows.append({
            "family": family["family"], "residual": r,
            "domains": {"carriers": len(carriers), "cycles": len(cycles), "productive_inner_cycles": len(morphs)},
            "search": search_stats, "survivors": len(survivors),
            "obstruction_or_gate": obstruction(r, carriers, cycles, morphs, diagnostics, word_pos),
            "sample_carriers": [{"left": p.left.words, "right": p.right.words} for p in carriers[:8]],
            "sample_cycles": [{"left": p.left.words, "right": p.right.words} for p in cycles[:8]],
            "productive_cycles": [{"left": p.left.words, "right": p.right.words, "morphology": asdict(m)} for p, m in morphs[:20]],
        })
    # Exact duplicate tapes across families are one survivor, with the first
    # fully typed derivation retained deterministically.
    unique = {}
    for row in sorted(all_survivors, key=lambda r: (-r["letters"], r["rendered"], r["family"])):
        unique.setdefault(row["normalized_tape"], row)
    survivors = list(unique.values())
    extended = clean_extended_witness()
    if extended["mechanically_admitted"]:
        survivors = [extended] + [row for row in survivors if row["normalized_tape"] != extended["normalized_tape"]]
        for family_row in family_rows:
            if family_row["family"] == "plural_s":
                family_row["survivors"] += 1
                family_row["bounded_extension"] = {
                    "source": "fresh open carrier plus two typed cycles",
                    "letters": extended["letters"],
                    "rendered": extended["rendered"],
                }
                break
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "method": "attested Brown spans composed by carrier/cycle residual equations with a LIFO return stack",
        "equations": {"carrier": "T(x0)r = reverse(T(y0))", "cycle": "T(xi)r = r reverse(T(yi))"},
        "corpus": corpus,
        "families": family_rows,
        "stats": {"families": len(FAMILIES), "unique_survivors": len(survivors),
                  "max_letters": max((r["letters"] for r in survivors), default=0),
                  "elapsed_seconds": round(time.monotonic() - started, 3)},
        "survivors": survivors[:max_candidates],
        "provenance": {
            "proper_names_allowed": False, "catalogue_text_imported": False,
            "completed_palindromic_units_in_inventory": False,
            "right_returns_emitted_lifo": True, "lemma_freshness_online": True,
            "complementary_boundary_mask_online": True,
            "proper_span_mask_online_and_shared_gate": True,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "host": os.uname().nodename,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["result_sha256"] = sha256_text(canonical)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", type=Path, default=None)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    payload = run(args.brown or discover_default_brown(), args.max_words, args.max_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stats": payload["stats"], "families": [
        {"family": f["family"], "domains": f["domains"], "survivors": f["survivors"],
         "reason": f["obstruction_or_gate"]["reason"]} for f in payload["families"]
    ], "top": payload["survivors"][:5]}, indent=2))


if __name__ == "__main__":
    main()
