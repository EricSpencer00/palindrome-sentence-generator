"""Jointly project authored prose onto a palindrome and move word boundaries.

Each symmetric pair of source positions supplies two possible target letters.
Choosing either realizes the unconstrained minimum Hamming repair for that
pair. Other letters spend an explicit extra-edit budget. Forward and reversed
dictionary tries independently segment the projected tape from its ends;
their unfinished words can join across the centre. No palindrome catalogue,
existing pair bank, or complete reversed clause is construction material.

This is a bounded, length-preserving repair experiment. Search ordering by
edit cost and source-word retention is NOT evidence of readability. Every
rendered proposal is preserved with a separate exact audit and shared gates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    ORDINARY_TWO_LETTER_WORDS,
    mechanical_admission_checks,
)


SEEDS = (
    ("editors-desserts", "Editors revise reports before sharing dessert.",
     "Stressed editors review rough drafts while patient colleagues propose clearer wording, hoping to finish revised reports before serving desserts."),
    ("walkers-spots", "Walkers use marked stops to find quiet places.",
     "Stops along woodland paths give tired walkers welcome shelter while local guides describe hidden ponds, distant villages, and peaceful picnic spots."),
    ("repair-strap", "Mechanics collect damaged bicycle parts for repair.",
     "Parts scattered beneath workshop benches remind careful mechanics which broken bicycles need fresh brakes, stronger wheels, or another leather strap."),
    ("sketch-ward", "An artist draws a quiet hospital scene.",
     "Draw several quiet nurses beside wide windows while young visitors arrange fresh flowers, share gentle stories, and brighten this crowded hospital ward."),
    ("fence-peek", "A visitor quietly observes a working garden.",
     "Keep curious visitors behind wooden fences while careful gardeners move heavy branches, repair damaged trellises, and permit one final quiet peek."),
    ("shelter-pets", "Visitors help settle animals at a shelter.",
     "Step through sheltered gardens where volunteers carry clean blankets, offer warm meals, arrange fresh bedding, and comfort newly rescued household pets."),
)

# Frozen local diagnostic: dictionary membership alone admitted abbreviations
# and isolated syllables (la/ni/fe/st/te/di) in the first run. This explicit
# small-word policy is conservative, not a complete definition of English.
def ordinary_short_forms(text: str) -> bool:
    words = re.findall("[a-z]+", text.lower())
    return all(len(word) != 2 or word in ORDINARY_TWO_LETTER_WORDS for word in words)


def independent_audit(text: str) -> dict:
    """Recompute ASCII tape and symmetry without using the constructor helpers."""
    chars = []
    unsupported = []
    for char in text:
        if "A" <= char <= "Z":
            chars.append(chr(ord(char) + 32))
        elif "a" <= char <= "z":
            chars.append(char)
        elif char.isalpha():
            unsupported.append(char)
    tape = "".join(chars)
    mismatches = [[i, len(tape) - i - 1] for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    return {"normalized": tape, "letters": len(tape),
            "supported_ascii": not unsupported,
            "exact": bool(tape) and not mismatches and not unsupported,
            "mismatch_pairs": mismatches}


def source_tape(text: str) -> str:
    return "".join(re.findall("[a-z]", text.lower()))


def make_trie(vocabulary: set[str], reverse: bool = False) -> dict:
    root: dict = {}
    for word in sorted(vocabulary):
        node = root
        for char in word[::-1] if reverse else word:
            node = node.setdefault(char, {})
        node[""] = word
    return root


def source_projection(text: str, side: str) -> str:
    """A visible complete proposal before lexical repair; may contain nonwords."""
    tape = list(source_tape(text))
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        tape[i] = tape[j] = tape[i] if side == "left" else tape[j]
    position = 0
    rendered = []
    for char in text:
        if char.isascii() and char.isalpha():
            letter = tape[position]
            rendered.append(letter.upper() if char.isupper() else letter)
            position += 1
        else:
            rendered.append(char)
    return "".join(rendered)


@dataclass(frozen=True)
class State:
    half: str = ""
    left_buffer: str = ""
    right_buffer: str = ""
    left_words: tuple[str, ...] = ()
    right_words: tuple[str, ...] = ()  # outside to inside
    extra_edits: int = 0
    source_word_letters: int = 0


def node_at(trie: dict, buffer: str) -> dict:
    for char in buffer:
        trie = trie[char]
    return trie


def lexical_options(node: dict, buffer: str, word: str | None,
                    used: frozenset[str]) -> list[tuple[str, str | None]]:
    options = [(buffer, None)] if any(key for key in node) else []
    if word and word not in used and word != word[::-1]:
        options.append(("", word))
    return options


def search(seed: str, vocabulary: set[str], *, beam: int = 256,
           extra_edits: int = 4, limit: int = 12) -> tuple[list[dict], dict]:
    """Bounded two-trie search through a coupled character-substitution lattice."""
    if beam < 1 or extra_edits < 0 or limit < 1:
        raise ValueError("beam and limit must be positive; extra edits nonnegative")
    tape = source_tape(seed)
    source_words = set(re.findall("[a-z]+", seed.lower()))
    forward, reverse = make_trie(vocabulary), make_trie(vocabulary, True)
    frontier = [State()]
    lower_bound = sum(tape[i] != tape[-i - 1] for i in range(len(tape) // 2))
    stats = {"source_letters": len(tape), "unconstrained_minimum_substitutions": lower_bound,
             "states_expanded": 0, "states_pruned_by_beam": 0,
             "edit_budget_rejections": 0, "duplicate_word_rejections": 0,
             "depths": [], "complete_lexical_closures": 0}
    for depth in range(len(tape) // 2):
        successors = []
        for state in frontier:
            stats["states_expanded"] += 1
            lnode = node_at(forward, state.left_buffer)
            rnode = node_at(reverse, state.right_buffer)
            available = sorted((lnode.keys() & rnode.keys()) - {""})
            used = frozenset(state.left_words + state.right_words)
            for char in available:
                cost = int(char != tape[depth]) + int(char != tape[-depth - 1])
                extra = cost - int(tape[depth] != tape[-depth - 1])
                if state.extra_edits + extra > extra_edits:
                    stats["edit_budget_rejections"] += 1
                    continue
                ln, rn = lnode[char], rnode[char]
                lo = lexical_options(ln, state.left_buffer + char, ln.get(""), used)
                ro = lexical_options(rn, state.right_buffer + char, rn.get(""), used)
                for (lb, lw), (rb, rw) in itertools.product(lo, ro):
                    if lw and lw == rw:
                        stats["duplicate_word_rejections"] += 1
                        continue
                    added = tuple(word for word in (lw, rw) if word)
                    successors.append(State(
                        state.half + char, lb, rb,
                        state.left_words + ((lw,) if lw else ()),
                        state.right_words + ((rw,) if rw else ()),
                        state.extra_edits + extra,
                        state.source_word_letters + sum(len(w) for w in added if w in source_words),
                    ))
        successors.sort(key=lambda s: (
            s.extra_edits, -s.source_word_letters,
            len(s.left_words) + len(s.right_words), s.half,
            s.left_words, s.right_words, s.left_buffer, s.right_buffer))
        stats["depths"].append({"paired_letters": depth + 1,
                                "successors_before_beam": len(successors)})
        stats["states_pruned_by_beam"] += max(0, len(successors) - beam)
        frontier = successors[:beam]
        if not frontier:
            break

    results = []
    seen = set()
    for state in frontier:
        if len(state.half) != len(tape) // 2:
            continue
        centre_options = "abcdefghijklmnopqrstuvwxyz" if len(tape) % 2 else ("",)
        for char in centre_options:
            cost = int(char != tape[len(tape) // 2]) if char else 0
            if state.extra_edits + cost > extra_edits:
                continue
            centre_word = state.left_buffer + char + state.right_buffer[::-1]
            if centre_word and (centre_word not in vocabulary or centre_word == centre_word[::-1]):
                continue
            words = state.left_words + ((centre_word,) if centre_word else ()) + state.right_words[::-1]
            if len(words) != len(set(words)):
                continue
            rendered = " ".join(words).capitalize() + "."
            if rendered in seen:
                continue
            seen.add(rendered)
            stats["complete_lexical_closures"] += 1
            target = source_tape(rendered)
            # Constructor assertions are supplemental: final records independently audit text.
            assert len(target) == len(tape)
            assert target == state.half + char + state.half[::-1]
            assert sum(a != b for a, b in zip(tape, target)) == lower_bound + state.extra_edits + cost
            results.append({"text": rendered, "extra_substitutions": state.extra_edits + cost,
                            "preserved_source_word_letters": state.source_word_letters +
                            (len(centre_word) if centre_word in source_words else 0)})
    results.sort(key=lambda row: (row["extra_substitutions"], -row["preserved_source_word_letters"], row["text"]))
    stats["rendered_closures_recorded"] = len(results)
    # Preserve every rendered closure; limit marks inspection order, not silent censoring.
    for i, row in enumerate(results):
        row["among_first_inspection_batch"] = i < limit
    return results, stats


def record(text: str, source_id: str, seed: str, kind: str) -> dict:
    audit = independent_audit(text)
    original = source_tape(seed)
    target = audit["normalized"]
    checks = mechanical_admission_checks(text, min_letters=100, max_letters=200)
    checks["independent_exact_audit"] = audit["exact"]
    checks["source_length_preserved"] = len(target) == len(original)
    checks["experiment_ordinary_short_forms"] = ordinary_short_forms(text)
    lengths = [len(w) for w in re.findall("[a-z]+", text.lower())]
    boundaries = list(itertools.accumulate(lengths))[:-1]
    shifted = sorted(set(boundaries) ^ {len(target) - b for b in boundaries})
    return {"source_id": source_id, "kind": kind, "text": text,
            "independent_audit": audit, "checks": checks,
            "rejection_codes": [key for key, value in checks.items() if not value],
            "character_substitutions": sum(a != b for a, b in zip(original, target)),
            "word_boundary_offsets": boundaries, "unmatched_reflected_boundaries": shifted,
            "reader_status": "No independent reader evidence; no readability or semantic-preservation claim."}


def run(*, beam: int, extra_edits: int, min_zipf: float,
        short_policy: str = "ordinary") -> dict:
    from experiments.bidirectional_attested_span_mining import common_lexicon
    vocabulary = {word for word in common_lexicon(min_zipf)
                  if len(word) > 1 and word != word[::-1] and word.isascii() and word.isalpha()}
    if short_policy == "ordinary":
        vocabulary = {word for word in vocabulary if ordinary_short_forms(word)}
    records, searches = [], []
    for identifier, event, seed in SEEDS:
        for side in ("left", "right"):
            records.append(record(source_projection(seed, side), identifier, seed,
                                  f"unlexicalized_{side}_projection"))
        rows, stats = search(seed, vocabulary, beam=beam, extra_edits=extra_edits)
        searches.append({"source_id": identifier, "authored_event": event,
                         "authored_sentence": seed, **stats})
        for row in rows:
            records.append(record(row["text"], identifier, seed, "joint_lexical_repair") | row)
    provenance = {"source_sentences": SEEDS, "vocabulary": sorted(vocabulary),
                  "ordinary_two_letter_forms": sorted(ORDINARY_TWO_LETTER_WORDS),
                  "construction_sources": "Six independently authored task-local sentences; common dictionary forms.",
                  "catalogue_usage": "Exclusion only through the unchanged shared admission gate."}
    return {"status": "complete_bounded_coupled_projection_experiment",
            "config": {"beam": beam, "extra_edits_above_hamming_lower_bound": extra_edits,
                       "minimum_zipf": min_zipf, "minimum_candidate_letters": 100,
                       "search_short_word_policy": short_policy},
            "provenance": provenance,
            "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
            "searches": searches, "records": records,
            "mechanically_eligible": [row for row in records if not row["rejection_codes"]],
            "readable_survivors": [],
            "scope": "Beam-limited dictionary repair; no exhaustive impossibility claim and no reader evidence.",
            "search_order_warning": "Edit cost and source-word retention only order the search; neither admits or rates text."}


def audit_artifact(path: Path) -> dict:
    """Recompute every rendered proposal without trusting saved admission flags."""
    source = json.loads(path.read_text())
    records = []
    for index, original in enumerate(source["records"]):
        text = original["text"]
        audit = independent_audit(text)
        words = re.findall("[a-z]+", text.lower())
        bad_short_words = sorted({word for word in words
                                  if len(word) == 2 and word not in ORDINARY_TWO_LETTER_WORDS})
        checks = mechanical_admission_checks(text, min_letters=100, max_letters=200)
        checks["independent_exact_audit"] = audit["exact"]
        checks["experiment_ordinary_short_forms"] = not bad_short_words
        records.append({"source_record_index": index, "source_id": original["source_id"],
                        "kind": original["kind"], "text": text,
                        "independent_audit": audit, "checks": checks,
                        "excluded_short_forms": bad_short_words,
                        "rejection_codes": [key for key, value in checks.items() if not value],
                        "reader_status": "No independent human review was performed."})
    return {"status": "complete_independent_rendered_proposal_audit",
            "source_artifact": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "ordinary_two_letter_forms": sorted(ORDINARY_TWO_LETTER_WORDS),
            "records": records, "records_total": len(records),
            "independently_exact": sum(row["independent_audit"]["exact"] for row in records),
            "rejected_by_short_form_diagnostic": sum(bool(row["excluded_short_forms"]) for row in records),
            "no_hard_gate_rejections": sum(not row["rejection_codes"] for row in records),
            "readable_survivors": [],
            "interpretation": "Short-form failures reject this construction output, not every possible use of those words. A pass would still establish no readability."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument("--extra-edits", type=int, default=4)
    parser.add_argument("--min-zipf", type=float, default=3.5)
    parser.add_argument("--short-policy", choices=("ordinary", "dictionary"), default="ordinary")
    parser.add_argument("--audit-source", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = (audit_artifact(args.audit_source) if args.audit_source else
              run(beam=args.beam, extra_edits=args.extra_edits, min_zipf=args.min_zipf,
                  short_policy=args.short_policy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "lexical_closures": sum(row["complete_lexical_closures"] for row in result.get("searches", [])),
                      "mechanically_eligible": len(result.get("mechanically_eligible", [])),
                      "readable_survivors": 0}, indent=2))


if __name__ == "__main__":
    main()
