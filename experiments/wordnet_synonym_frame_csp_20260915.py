"""WordNet synonym-frame CSP for genuinely new palindrome constructions.

The construction unit is a semantic dependency frame, not a mirrored word
list.  Each slot carries a concept and is lexicalized independently on the
left and right by WordNet lemmas.  A character CSP joins complete frame yields
only when the right yield is the exact reverse tape of the left yield.  The
frames are reparsed as ordinary clauses after joining; no source sentence is
copied and no output is reader evidence until a blinded study is run.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from itertools import product
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/wordnet-synonym-frame-csp-20260915.json"
EXPERIMENT_ID = "wordnet-synonym-frame-csp-20260915"
SIGNATURE = (
    "wordnet-synonym-frame-csp|dependency-slot-paraphrase-lattice|"
    "semantic-preserving-lexical-choice|character-equation-constraint|"
    "independent-reader-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Option:
    word: str
    concept: str
    synsets: tuple[str, ...]
    zipf: float


# The concepts deliberately describe roles rather than an existing sentence.
# WordNet is used to provide independent lexical alternatives for each concept.
CONCEPT_SEEDS: dict[str, tuple[str, str]] = {
    "person": ("person", "n"),
    "worker": ("worker", "n"),
    "document": ("document", "n"),
    "message": ("message", "n"),
    "plan": ("plan", "n"),
    "book": ("book", "n"),
    "read": ("read", "v"),
    "write": ("write", "v"),
    "send": ("send", "v"),
    "make": ("make", "v"),
    "keep": ("keep", "v"),
    "see": ("see", "v"),
    "mark": ("mark", "v"),
    "report": ("report", "v"),
    "calm": ("calm", "a"),
    "clear": ("clear", "a"),
    "ready": ("ready", "a"),
    "kind": ("kind", "a"),
    "good": ("good", "a"),
    "quietly": ("quietly", "r"),
    "carefully": ("carefully", "r"),
}

# Each frame is a complete, grammatical clause shape.  A frame's slots carry
# concepts; the same frame may be lexicalized differently on either side.
FRAMES: dict[str, tuple[str, ...]] = {
    "pron_action_object": ("pronoun", "action", "det", "document"),
    "pron_report_object": ("pronoun", "report_action", "det", "message"),
    "agent_action_object": ("det", "person", "action3", "det", "document"),
    "agent_make_plan": ("det", "worker", "action3", "det", "plan"),
    "agent_adverb_action": ("det", "person", "adverb", "action3", "det", "book"),
    "agent_copular_adj": ("det", "person", "copula", "adjective"),
}

FUNCTION_WORDS = frozenset(
    "a an the some one this that my our your their i we you they he she it "
    "is are was were be do does did can and or but if as of to in on at by "
    "for with from near over under".split()
)


def _clean(word: str) -> str | None:
    word = word.casefold().replace("_", " ")
    if not re.fullmatch(r"[a-z]{2,14}", word):
        return None
    return word


def _wordnet_options() -> dict[str, list[Option]]:
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    out: dict[str, list[Option]] = {}
    for concept, (seed, pos) in CONCEPT_SEEDS.items():
        by_word: dict[str, set[str]] = {}
        for synset in wn.synsets(seed, pos=pos):
            for lemma in synset.lemmas():
                word = _clean(lemma.name())
                if word is None or zipf_frequency(word, "en") < 3.0:
                    continue
                by_word.setdefault(word, set()).add(synset.name())
        # Keep the seed even where WordNet has no useful high-frequency lemma.
        if _clean(seed):
            by_word.setdefault(seed, set()).add(f"{seed}.{pos}.seed")
        options = [Option(w, concept, tuple(sorted(s)), zipf_frequency(w, "en"))
                   for w, s in by_word.items()]
        options.sort(key=lambda row: (-row.zipf, row.word))
        out[concept] = options[:28]

    # Grammatical function slots and inflectional variants are explicit rather
    # than WordNet synonyms; they are not construction units.
    out["pronoun"] = [Option(w, "pronoun", (), 7.0)
                      for w in "i we you they he she".split()]
    out["det"] = [Option(w, "det", (), 7.0)
                  for w in "a an the some one this that my our your their".split()]
    out["copula"] = [Option(w, "copula", (), 7.0)
                     for w in "is are was were".split()]
    # Frame-level semantic classes are unions of the narrower WordNet seed
    # inventories.  Keeping the originating concept on each option preserves
    # provenance while allowing a dependency slot such as ACTION to choose
    # among read/write/send alternatives.
    def union(name: str, concepts: tuple[str, ...]) -> None:
        by_word: dict[str, Option] = {}
        for concept in concepts:
            for option in out.get(concept, ()):
                prior = by_word.get(option.word)
                if prior is None or option.zipf > prior.zipf:
                    by_word[option.word] = option
        out[name] = sorted(by_word.values(), key=lambda row: (-row.zipf, row.word))[:42]

    union("action", ("read", "write", "send", "make", "keep", "see", "mark"))
    union("report_action", ("report", "send", "write", "mark"))
    union("adjective", ("calm", "clear", "ready", "kind", "good"))
    out["adverb"] = [Option(w, "adverb", (), 6.0)
                     for w in "carefully quietly clearly calmly well".split()]
    # Present-tense forms are used with third-person subjects.  The base
    # WordNet alternatives remain linked to their concept and receive a
    # deterministic -s/-es inflection for the frame.
    for source, target in (("action", "action3"), ("report_action", "report3")):
        variants: list[Option] = []
        for option in out[source]:
            suffix = "es" if option.word.endswith(("s", "x", "z", "ch", "sh")) else "s"
            variants.append(Option(option.word + suffix, option.concept,
                                   option.synsets, option.zipf - 0.2))
        out[target] = variants
    return out


def _slots(options: dict[str, list[Option]], frame: tuple[str, ...]) -> list[list[Option]]:
    return [options.get(slot, []) for slot in frame]


def _match_target(target: str, choices: list[list[Option]], *, limit: int = 12) -> list[tuple[Option, ...]]:
    """Segment a fixed reverse tape into a complete frame yield."""
    results: list[tuple[Option, ...]] = []

    def visit(index: int, slot: int, chosen: tuple[Option, ...]) -> None:
        if len(results) >= limit:
            return
        if slot == len(choices):
            if index == len(target):
                results.append(chosen)
            return
        for option in choices[slot]:
            if target.startswith(option.word, index):
                visit(index + len(option.word), slot + 1, chosen + (option,))

    visit(0, 0, ())
    return results


def _distinct_content(words: tuple[Option, ...]) -> bool:
    content = [option.word for option in words if option.word not in FUNCTION_WORDS]
    return len(content) == len(set(content))


def _audit(text: str, left: tuple[Option, ...], right: tuple[Option, ...]) -> dict:
    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=240)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_tape": tape,
        "independent_ascii_tape": independent,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_exact": bool(independent) and independent == independent[::-1],
        "tapes_equal": tape == independent,
        "two_pointer_exact": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "left_synsets": [list(option.synsets) for option in left],
        "right_synsets": [list(option.synsets) for option in right],
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run(*, per_pair_limit: int = 400, match_limit: int = 8) -> dict:
    options = _wordnet_options()
    frame_names = list(FRAMES)
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    # The left side is enumerated in semantic slot order.  The right side is
    # solved against its required reverse tape, so no reflected word sequence
    # is ever emitted directly.
    for left_name in frame_names:
        left_choices = _slots(options, FRAMES[left_name])
        for right_name in frame_names:
            right_choices = _slots(options, FRAMES[right_name])
            checked = 0
            for left in product(*left_choices):
                if checked >= per_pair_limit:
                    break
                checked += 1
                stats["left_assignments"] += 1
                left_words = tuple(option.word for option in left)
                left_tape = "".join(left_words)
                if len(left_tape) * 2 < MIN_LETTERS:
                    continue
                matches = _match_target(left_tape[::-1], right_choices, limit=match_limit)
                stats["reverse_frame_calls"] += 1
                if not matches:
                    continue
                stats["reverse_frame_hits"] += len(matches)
                for right in matches:
                    words = left + right
                    if not _distinct_content(words):
                        stats["duplicate_content_reject"] += 1
                        continue
                    text = " ".join(option.word for option in words).capitalize() + "."
                    tape = normalize_letters(text)
                    if len(tape) < MIN_LETTERS or tape in seen:
                        continue
                    seen.add(tape)
                    audit = _audit(text, left, right)
                    row = {
                        "rendered": text,
                        "left_frame": left_name,
                        "right_frame": right_name,
                        "left_concepts": list(FRAMES[left_name]),
                        "right_concepts": list(FRAMES[right_name]),
                        "left_words": list(left_words),
                        "right_words": [option.word for option in right],
                        "audit": audit,
                        "reader_status": "not_run; programmatic diagnostics do not certify readability",
                    }
                    rows.append(row)
                    stats["exact"] += int(audit["exact"])
                    stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
            # Preserve at most a few actual boundary probes for every pair.
            if len(probes) < 160:
                for left in product(*left_choices):
                    left_tape = "".join(option.word for option in left)
                    if len(left_tape) * 2 < MIN_LETTERS:
                        continue
                    matches = _match_target(left_tape[::-1], right_choices, limit=1)
                    if matches:
                        probes.append({
                            "left_frame": left_name,
                            "right_frame": right_name,
                            "left_words": [option.word for option in left],
                            "required_reverse_tape": left_tape[::-1],
                            "right_match": [option.word for option in matches[0]],
                            "status": "candidate_checked",
                        })
                        break
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "WordNet synonym alternatives are attached to semantic dependency "
            "slots; each left frame yield is matched against an independently "
            "lexicalized right frame under an exact character equation."
        ),
        "novelty_preflight": {
            "registry_entries_before_run": 84,
            "excluded_routes_before_run": 6,
            "status": "formal_preflight_before_execution",
            "signature_overlap": [],
            "manual_review_required": False,
        },
        "config": {
            "frame_count": len(FRAMES),
            "frame_pairs": len(FRAMES) ** 2,
            "per_pair_left_assignment_limit": per_pair_limit,
            "reverse_match_limit": match_limit,
            "wordnet_synsets": "nltk.corpus.wordnet",
            "wordnet_downloaded": True,
            "catalogue_text_imported": False,
            "known_palindromes_imported": False,
        },
        "stats": {
            **dict(stats),
            "rendered_candidates": len(rows),
            "reader_eligible": 0,
        },
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "WordNet lemma inventories plus wordfreq frequency floor",
            "source_sentences_copied": False,
            "independent_exact_audits": ["normalized-tape-reversal", "ASCII-tape-reversal", "two-pointer"],
            "readability_certificate": False,
        },
        "next_repair": (
            "Carry agreement and subcategorization features through the same "
            "synonym lattice, then add an inflectional realization layer; this "
            "is a new repair to semantic-slot closure, not a reservoir or "
            "reverse-segmentation replay."
        ),
        "reader_gate": (
            "No row is reader evidence. Any mechanically admitted row must be "
            "manually screened for intact prose and frozen with randomized "
            "intact/shuffled controls before readers see it."
        ),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT}")
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
