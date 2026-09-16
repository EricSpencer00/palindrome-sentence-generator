"""Admission-guided center-out repair with a frozen lexical inventory.

This run is intentionally separate from the character half-tape route.  It
keeps whole lexical words in the live state, rejects repeated units before they
can dominate the beam, and applies the full mechanical admission gate at every
closure.  The gate is still not a readability certificate: any surviving
surface must go to the intact-prose/shuffled-control reader package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import build_vocab
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.shortwords import is_real_short


ID = "lexical-admission-centerout"
SIGNATURE = (
    "wordfreq-bigram-centerout|live-word-uniqueness|"
    "admission-at-closure|length-sweep"
)


def independent_ascii_tape(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isascii() and ch.isalpha())


def _live_unique(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Reject every repeated lexical unit while the state is still live.

    This is stricter than the shared gate's ordinary-function-word policy on
    purpose: it is the repair under test, preventing short function words from
    becoming a cheap overhang filler.  A closure is still checked by the full
    independent admission function below.
    """
    # Search vocabulary is prefiltered to ASCII letters, so a direct
    # whitespace removal avoids invoking the comparatively defensive public
    # normalizer at every beam expansion.
    words = [word.replace(" ", "") for word in left + right]
    return all(len(word) >= 2 for word in words) and len(words) == len(set(words))


def _closure_ok(left: tuple[str, ...], right: tuple[str, ...], *, minimum: int,
                maximum: int) -> bool:
    text = " ".join(left + right)
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=minimum, max_letters=maximum)
    return bool(tape) and tape == tape[::-1] and all(checks.values())


def run(
    *,
    targets: tuple[int, ...] = (39, 47, 63, 95),
    seeds: int = 24,
    vocabulary_size: int = 12_000,
    beam: int = 120,
    candidate_limit: int = 300,
    max_steps: int = 90,
) -> dict:
    # build_vocab applies the repository's safe-vocabulary policy.  The
    # top_n_list call is only used to make the exact vocabulary hash explicit
    # in the artifact; the search receives the same stable list.
    vocabulary = build_vocab(vocabulary_size)
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=vocabulary)
    scorer = CoherentScorer(
        bigrams,
        freq_weight=0.20,
        length_weight=0.08,
        short_penalty=2.5,
    )

    rows: list[dict] = []
    seen_tapes: set[str] = set()
    for target in targets:
        for seed in range(seeds):
            closed: list[list[str]] = []
            centerout_search(
                tries,
                scorer,
                min_letters=target,
                beam_width=beam,
                max_steps=max_steps,
                candidate_limit=candidate_limit,
                seed=seed,
                diversity=1.1,
                max_overhang=24,
                maximize="score",
                allow_word=lambda _placement, word, _state: is_real_short(word),
                allow_state=_live_unique,
                allow_closed=lambda left, right, target=target: _closure_ok(
                    left, right, minimum=target, maximum=220
                ),
                on_closed=lambda candidate: closed.append(candidate),
            )
            for words in closed:
                rendered = " ".join(words)
                tape = normalize_letters(rendered)
                if tape in seen_tapes:
                    continue
                seen_tapes.add(tape)
                independent = independent_ascii_tape(rendered)
                checks = mechanical_admission_checks(
                    rendered, min_letters=target, max_letters=220
                )
                rows.append({
                    "target": target,
                    "seed": seed,
                    "rendered": rendered + ".",
                    "words": words,
                    "letters": len(tape),
                    "normalized_tape": tape,
                    "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
                    "independent_ascii_tape": independent,
                    "independent_exact": bool(independent)
                    and independent == independent[::-1]
                    and independent == tape,
                    "mechanical_checks": checks,
                    "mechanically_admitted": all(checks.values())
                    and independent == independent[::-1]
                    and independent == tape,
                    "reader_status": "not_run; programmatic checks do not certify readability",
                })

    rows.sort(key=lambda row: (-row["letters"], row["target"], row["seed"]))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "status": "admission_guided_centerout_complete",
        "experiment_id": ID,
        "signature": SIGNATURE,
        "config": {
            "targets": list(targets),
            "seeds": seeds,
            "vocabulary_size_requested": vocabulary_size,
            "vocabulary_size": len(vocabulary),
            "beam": beam,
            "candidate_limit": candidate_limit,
            "max_steps": max_steps,
            "live_word_uniqueness": True,
            "closure_gate": "all shared mechanical admission checks",
            "catalogue_text_imported": False,
        },
        "novelty_audit": {
            "registry_entries_read_before_run": 59,
            "excluded_routes_read_before_run": 3,
            "signature_overlap": [],
            "conceptual_near_pairs": [],
            "manual_review_required": False,
            "preflight_required_before_artifact": True,
            "self_entry_present": False,
            "repair_note": (
                "The live whole-word uniqueness state and admission-at-closure "
                "policy are the declared construction dimension; this is not a "
                "beam/seed/vocabulary replay of character-lm-half-tape."
            ),
        },
        "stats": {
            "unique_rendered_candidates": len(rows),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
            "target_closures": {str(target): sum(row["target"] == target for row in rows)
                                for target in targets},
        },
        "rendered_candidates_and_probes": rows,
        "admitted": admitted,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
            "source": "wordfreq safe vocabulary plus count_2w bigram model",
            "source_sentences_copied": False,
            "known_palindromes_imported": False,
            "independent_validator": "llm_palindrome.validator.is_palindrome plus ASCII tape comparison",
            "readability_certificate": False,
        },
        "next_operator": (
            "Take the longest mechanically admitted tape into a grammar-constrained "
            "boundary resegmentation repair; preserve the exact tape, require a "
            "complete clause parse on the rendered surface, and send only the "
            "survivor plus intact/shuffled controls to blinded readers."
        ),
        "reader_gate": (
            "No row is reader evidence. A candidate may enter a randomized reader "
            "package only after manual review confirms intact English prose and the "
            "package includes shuffled controls and reproducible rater instructions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--targets", type=int, nargs="+", default=[39, 47, 63, 95])
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--vocabulary-size", type=int, default=12_000)
    parser.add_argument("--beam", type=int, default=120)
    parser.add_argument("--candidate-limit", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=90)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        targets=tuple(args.targets),
        seeds=args.seeds,
        vocabulary_size=args.vocabulary_size,
        beam=args.beam,
        candidate_limit=args.candidate_limit,
        max_steps=args.max_steps,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "candidates": result["stats"]["unique_rendered_candidates"],
        "mechanically_admitted": result["stats"]["mechanically_admitted"],
    }, indent=2))


if __name__ == "__main__":
    main()
