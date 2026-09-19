"""Collect exact phrase-lattice closures, then rerank them with a local LM.

The language model is a post-hoc ordering signal only.  It never supplies
letters, relaxes the character equation, or certifies readability.  The
construction is a center-out exact search over independently indexed word and
short-phrase units; every surfaced row is rechecked by the shared mechanical
gate and by an outside-in audit before the LM sees it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bidirectional_attested_span_mining import common_lexicon
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.validator import is_palindrome
from wordfreq import zipf_frequency


EXPERIMENT_ID = "phrase-lattice-gpt2-rerank-20260919"
SIGNATURE = (
    "center-out-exact-phrase-lattice|posthoc-distilgpt2-rerank|"
    "independent-pointer-sha-audit|mechanical-gate|reader-required"
)


def novelty_preflight() -> dict:
    """Fail closed if this exact construction signature already exists.

    The repository contains many deliberately different palindrome probes.
    This lane is useful only if it records a new search geometry, so the
    preflight scans prior experiment source rather than silently repeating a
    larger beam sweep under a new filename.
    """
    collisions: list[str] = []
    marker = "center-out-exact-phrase-lattice"
    for path in sorted((ROOT / "experiments").glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        try:
            source = path.read_text()
        except OSError:
            continue
        if SIGNATURE in source or marker in source:
            collisions.append(str(path.relative_to(ROOT)))
    return {
        "status": "passed" if not collisions else "blocked_duplicate_signature",
        "signature": SIGNATURE,
        "collisions": collisions,
        "search_geometry": (
            "exact center-out word/short-phrase lattice with a post-hoc local "
            "LM ordering; the LM never supplies characters or closes a state"
        ),
        "duplicate_sweep_rejected": bool(collisions),
    }


def audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [
        {"left": i, "right": len(tape) - 1 - i,
         "left_char": tape[i], "right_char": tape[-1 - i]}
        for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]
    ]
    return {
        "letters": len(tape),
        "normalized": tape,
        "two_pointer_exact": not mismatches and bool(tape),
        "validator_exact": is_palindrome(text),
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
        "mismatches": mismatches[:8],
    }


def build_units(min_zipf: float, phrase_limit: int) -> tuple[list[str], set[str]]:
    vocab = common_lexicon(min_zipf)
    phrases = build_inventory(
        str(ROOT / "data" / "count_2w.txt"),
        vocab=vocab,
        top_n=phrase_limit,
        min_count=3,
    )
    # Phrase units are proposals, not copied sentence surfaces.  Their inner
    # joins are scored as observed inventory edges, while the final surface is
    # still assembled by the character lattice.
    phrases = {
        phrase for phrase in phrases
        if 2 <= len(phrase.split()) <= 4
        and 7 <= len(phrase.replace(" ", "")) <= 24
    }
    units = sorted(
        set(vocab) | phrases,
        key=lambda unit: (-zipf_frequency(unit.split()[0], "en"), unit),
    )
    return units, vocab


def collect(*, seeds: int, beam: int, candidate_limit: int,
            phrase_limit: int, min_zipf: float) -> dict:
    preflight = novelty_preflight()
    if preflight["status"] != "passed":
        return {
            "experiment_id": EXPERIMENT_ID,
            "signature": SIGNATURE,
            "status": "preflight_blocked",
            "novelty_preflight": preflight,
            "rows": [],
            "closed_count": 0,
            "exact_count": 0,
            "mechanically_admitted_count": 0,
        }
    units, vocab = build_units(min_zipf, phrase_limit)
    tries = WordTries(units)
    bigrams = BigramModel.from_file(
        str(ROOT / "data" / "count_2w.txt"), vocab=vocab
    )
    scorer = CoherentScorer(
        bigrams,
        freq_weight=0.10,
        length_weight=0.13,
        phrase_weight=8.0,
        long_bonus=2.0,
        short_penalty=4.0,
        unit_bonus={
            unit: 4.0 + 1.5 * (len(unit.split()) - 1)
            for unit in units if " " in unit
        },
    )
    closed: dict[str, dict] = {}
    for seed in range(seeds):
        def remember(sequence: list[str], seed: int = seed) -> None:
            text = " ".join(sequence).strip()
            if not text:
                return
            tape = normalize_letters(text)
            if not 39 <= len(tape) <= 140:
                return
            row = closed.setdefault(text, {
                "rendered": text,
                "seed": seed,
                "units": sequence,
                "audit": audit(text),
            })
            row["seeds"] = sorted(set(row.get("seeds", [])) | {seed})

        centerout_search(
            tries,
            scorer,
            min_letters=39,
            beam_width=beam,
            max_steps=180,
            candidate_limit=candidate_limit,
            seed=seed,
            diversity=1.4,
            max_overhang=28,
            on_closed=remember,
        )

    rows = list(closed.values())
    for row in rows:
        row["mechanical_checks"] = mechanical_admission_checks(
            row["rendered"], min_letters=39, max_letters=140
        )
        row["mechanically_admitted"] = all(row["mechanical_checks"].values())
        row["reader_status"] = "not_run; human readers required"
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "closures_collected_pending_lm_rerank",
        "novelty_preflight": preflight,
        "config": {
            "seeds": seeds,
            "beam": beam,
            "candidate_limit": candidate_limit,
            "phrase_limit": phrase_limit,
            "min_zipf": min_zipf,
            "length_band": [39, 140],
        },
        "unit_count": len(units),
        "vocabulary_count": len(vocab),
        "closed_count": len(rows),
        "exact_count": len(exact),
        "mechanically_admitted_count": len(admitted),
        "rows": rows,
        "provenance": {
            "source": "local lexical and observed short-phrase inventory",
            "finished_sentence_reversal": False,
            "catalogue_text_imported": False,
            "lm_used_during_search": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }


def rerank(report: dict, model_name: str) -> dict:
    from llm_palindrome.lm_scoring import GPT2Scorer

    rows = [row for row in report["rows"] if row["mechanically_admitted"]]
    if not rows:
        report["rerank"] = {"model": model_name, "status": "no_admitted_rows", "rows": []}
        return report
    scorer = GPT2Scorer(model_name=model_name)
    details = scorer.score_details([row["rendered"] for row in rows], batch_size=8)
    ranked = []
    for row, detail in zip(rows, details):
        ranked.append({
            "rendered": row["rendered"],
            "letters": row["audit"]["letters"],
            "lm_detail": detail,
            "audit": row["audit"],
            "mechanical_checks": row["mechanical_checks"],
            "reader_status": row["reader_status"],
        })
    ranked.sort(key=lambda row: (row["lm_detail"]["per_letter"], row["letters"]), reverse=True)
    report["status"] = "posthoc_lm_rerank_complete"
    report["rerank"] = {
        "model": model_name,
        "search_uses_feedback": False,
        "readability_certified": False,
        "human_reader_gate_required": True,
        "rows": ranked,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--candidate-limit", type=int, default=700)
    parser.add_argument("--phrase-limit", type=int, default=30_000)
    parser.add_argument("--min-zipf", type=float, default=3.0)
    parser.add_argument("--model", default="distilgpt2")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    report = collect(
        seeds=args.seeds,
        beam=args.beam,
        candidate_limit=args.candidate_limit,
        phrase_limit=args.phrase_limit,
        min_zipf=args.min_zipf,
    )
    report = rerank(report, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "closed": report["closed_count"],
        "exact": report["exact_count"],
        "admitted": report["mechanically_admitted_count"],
        "reranked": len(report.get("rerank", {}).get("rows", [])),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
