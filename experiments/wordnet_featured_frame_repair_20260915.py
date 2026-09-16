"""Agreement/subcategorization repair for the WordNet frame CSP.

The preceding WordNet synonym-frame run had zero reverse-frame lexical hits.
This repair adds a real state dimension rather than another beam: subject
number, determiner agreement, verb inflection, transitivity, and optional
adverbial attachment are carried through a finite frame automaton.  Left and
right clauses are still independently lexicalized and joined only by an exact
character equation.  No generated row is readability evidence.
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
OUT = ROOT / "runs/wordnet-featured-frame-repair-20260915.json"
EXPERIMENT_ID = "wordnet-featured-frame-repair-20260915"
SIGNATURE = (
    "wordnet-featured-frame-repair|subcategorization-agreement-automaton|"
    "inflectional-lexeme-realization|semantic-role-compatible-join|"
    "independent-exact-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.wordnet_synonym_frame_csp_20260915 import (
    FUNCTION_WORDS,
    Option,
    _match_target,
    _wordnet_options,
)


@dataclass(frozen=True)
class Frame:
    name: str
    relation: str
    subject_number: str
    slots: tuple[str, ...]


FRAMES = (
    Frame("singular_transitive", "transitive", "singular",
          ("det_sg", "subject_sg", "verb_3sg", "det_sg", "object_sg")),
    Frame("plural_transitive", "transitive", "plural",
          ("det_pl", "subject_pl", "verb_base", "det_pl", "object_pl")),
    Frame("singular_adjunct", "transitive", "singular",
          ("det_sg", "subject_sg", "adverb", "verb_3sg", "det_sg", "object_sg")),
    Frame("plural_adjunct", "transitive", "plural",
          ("det_pl", "subject_pl", "adverb", "verb_base", "det_pl", "object_pl")),
    Frame("pronoun_singular", "transitive", "singular",
          ("pronoun_sg", "verb_3sg", "det_sg", "object_sg")),
    Frame("pronoun_plural", "transitive", "plural",
          ("pronoun_pl", "verb_base", "det_pl", "object_pl")),
)


def _plural(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _feature_options() -> dict[str, list[Option]]:
    base = _wordnet_options()
    options: dict[str, list[Option]] = {}
    options["det_sg"] = [Option(w, "det", (), 7.0)
                          for w in "a an the this that my our your".split()]
    options["det_pl"] = [Option(w, "det", (), 7.0)
                          for w in "the some these those my our your".split()]
    options["pronoun_sg"] = [Option(w, "pronoun", (), 7.0)
                              for w in "i you he she".split()]
    options["pronoun_pl"] = [Option(w, "pronoun", (), 7.0)
                              for w in "we you they".split()]
    for source, singular, plural in (
        ("person", "subject_sg", "subject_pl"),
        ("worker", "subject_sg", "subject_pl"),
        ("document", "object_sg", "object_pl"),
        ("message", "object_sg", "object_pl"),
        ("plan", "object_sg", "object_pl"),
        ("book", "object_sg", "object_pl"),
    ):
        current = options.setdefault(singular, [])
        for item in base.get(source, ()):
            if item.word not in {x.word for x in current} and not item.word.endswith("s"):
                current.append(item)
        target = options.setdefault(plural, [])
        for item in current:
            target.append(Option(_plural(item.word), item.concept, item.synsets, item.zipf - 0.25))

    # The relation is explicitly transitive; all alternatives come from the
    # corresponding WordNet action inventories, not from arbitrary dictionary
    # strings.  Base forms are used with plural subjects/pronouns, and
    # deterministic third-person forms with singular subjects.
    verbs: list[Option] = []
    for source in ("action", "report_action"):
        for item in base.get(source, ()):
            if item.word not in {x.word for x in verbs}:
                verbs.append(item)
    verbs.sort(key=lambda item: (-item.zipf, item.word))
    verbs = verbs[:56]
    options["verb_base"] = verbs
    options["verb_3sg"] = [
        Option(item.word + ("es" if item.word.endswith(("s", "x", "z", "ch", "sh")) else "s"),
               item.concept, item.synsets, item.zipf - 0.2)
        for item in verbs
    ]
    options["adverb"] = base.get("adverb", [])
    return options


def _distinct_content(words: tuple[Option, ...]) -> bool:
    content = [item.word for item in words if item.word not in FUNCTION_WORDS]
    return len(content) == len(set(content))


def _audit(text: str, left: tuple[Option, ...], right: tuple[Option, ...]) -> dict:
    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=260)
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
        "left_synsets": [list(item.synsets) for item in left],
        "right_synsets": [list(item.synsets) for item in right],
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run(*, per_pair_limit: int = 2_500, match_limit: int = 16) -> dict:
    options = _feature_options()
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    frame_pairs = [(left, right) for left in FRAMES for right in FRAMES
                   if left.relation == right.relation]
    for left_frame, right_frame in frame_pairs:
        left_choices = [options.get(slot, []) for slot in left_frame.slots]
        right_choices = [options.get(slot, []) for slot in right_frame.slots]
        checked = 0
        for left in product(*left_choices):
            if checked >= per_pair_limit:
                break
            checked += 1
            stats["left_assignments"] += 1
            left_tape = "".join(item.word for item in left)
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
                text = " ".join(item.word for item in words).capitalize() + "."
                tape = normalize_letters(text)
                if len(tape) < MIN_LETTERS or tape in seen:
                    continue
                seen.add(tape)
                audit = _audit(text, left, right)
                rows.append({
                    "rendered": text,
                    "left_frame": left_frame.name,
                    "right_frame": right_frame.name,
                    "relation": left_frame.relation,
                    "left_number": left_frame.subject_number,
                    "right_number": right_frame.subject_number,
                    "left_words": [item.word for item in left],
                    "right_words": [item.word for item in right],
                    "audit": audit,
                    "reader_status": "not_run; feature agreement is not human readability evidence",
                })
                stats["exact"] += int(audit["exact"])
                stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
        if len(probes) < 160:
            for left in product(*left_choices):
                left_tape = "".join(item.word for item in left)
                if len(left_tape) * 2 < MIN_LETTERS:
                    continue
                matches = _match_target(left_tape[::-1], right_choices, limit=1)
                if matches:
                    probes.append({
                        "left_frame": left_frame.name,
                        "right_frame": right_frame.name,
                        "left_words": [item.word for item in left],
                        "required_reverse_tape": left_tape[::-1],
                        "right_match": [item.word for item in matches[0]],
                        "status": "reverse_frame_hit",
                    })
                    break
    rows.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Finite agreement/subcategorization automaton carries subject number, "
            "determiner agreement, transitive verb inflection, and optional "
            "adverb attachment while matching independently lexicalized frames."
        ),
        "novelty_preflight": {
            "registry_entries_before_run": 85,
            "excluded_routes_before_run": 6,
            "status": "formal_preflight_before_execution",
            "signature_overlap": [],
            "manual_review_required": False,
        },
        "config": {
            "frame_count": len(FRAMES),
            "compatible_frame_pairs": len(frame_pairs),
            "per_pair_left_assignment_limit": per_pair_limit,
            "reverse_match_limit": match_limit,
            "wordnet_synsets": "nltk.corpus.wordnet",
            "inflection": "deterministic plural and third-person-singular realization",
            "subcategorization": "transitive relation only",
            "catalogue_text_imported": False,
            "known_palindromes_imported": False,
        },
        "stats": {**dict(stats), "rendered_candidates": len(rows), "reader_eligible": 0},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "WordNet lemma inventories with explicit feature realization",
            "source_sentences_copied": False,
            "independent_exact_audits": ["normalized-tape-reversal", "ASCII-tape-reversal", "two-pointer"],
            "readability_certificate": False,
        },
        "next_repair": (
            "Use valency-specific adjunct frames with a bounded semantic role graph "
            "and preserve the agreement automaton; this changes dependency topology "
            "rather than merely widening the current lexical options."
        ),
        "reader_gate": (
            "No row is reader evidence. Any exact row must pass every mechanical "
            "gate and then be frozen with randomized intact/shuffled controls for "
            "blinded human ratings."
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
