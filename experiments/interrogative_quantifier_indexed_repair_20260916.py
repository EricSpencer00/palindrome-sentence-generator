"""Target-indexed repair of the interrogative/quantifier FSM.

The preceding family built every answer-frame product before asking whether it
could match a reverse tape.  This repair keeps the same inversion and
question/answer dependency states, but walks each answer slot only when its
lexical alternatives match the concrete target prefix.  It is a repair of that
family, not a new lexical bank or a larger beam.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/interrogative-quantifier-indexed-repair-20260916.json"
EXPERIMENT_ID = "interrogative-quantifier-indexed-repair-20260916"
SIGNATURE = (
    "interrogative-quantifier-fsm|target-tape-indexed-join|"
    "slot-prefix-automaton|question-answer-dependency|independent-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.interrogative_quantifier_fsm_20260915 import FRAMES, OPTIONS, _valid

KNOWN = {
    normalize_letters(x)
    for x in json.loads((ROOT / "data/known_palindromes.json").read_text())
}


def _audit(text: str) -> dict:
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
        "two_pointer_exact": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def _target_matches(slots: tuple[str, ...], target: str, stats: Counter) -> list[tuple[str, ...]]:
    """Return only slot assignments whose concatenation is exactly *target*."""
    by_slot: dict[str, tuple[str, ...]] = {
        slot: tuple(OPTIONS[slot]) for slot in set(slots)
    }

    @lru_cache(maxsize=50_000)
    def visit(index: int, offset: int) -> tuple[tuple[str, ...], ...]:
        if index == len(slots):
            return ((),) if offset == len(target) else ()
        slot = slots[index]
        found: list[tuple[str, ...]] = []
        for word in by_slot[slot]:
            letters = normalize_letters(word)
            if not target.startswith(letters, offset):
                stats["slot_prefix_trials"] += 1
                continue
            stats["slot_prefix_hits"] += 1
            for tail in visit(index + 1, offset + len(letters)):
                found.append((word,) + tail)
                if len(found) >= 12:
                    return tuple(found)
        return tuple(found)

    return list(visit(0, 0))


def run(*, max_left: int = 140_000) -> dict:
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    for frame_name, frame in FRAMES.items():
        left_options = [OPTIONS[slot] for slot in frame["left"]]
        checked = 0
        for left in itertools.product(*left_options):
            if checked >= max_left:
                break
            checked += 1
            stats["left_assignments_checked"] += 1
            if not _valid(left):
                stats["left_feature_rejects"] += 1
                continue
            left_tape = "".join(left)
            if len(left_tape) * 2 + 1 < MIN_LETTERS:
                stats["short_left_rejects"] += 1
                continue
            for center in "abcdefghijklmnopqrstuvwxyz":
                stats["center_equations"] += 1
                target = center + left_tape[::-1]
                matches = _target_matches(tuple(frame["right"]), target, stats)
                stats["target_index_lookups"] += 1
                if not matches:
                    stats["target_index_misses"] += 1
                    continue
                stats["target_index_hits"] += len(matches)
                for right in matches:
                    rendered = " ".join(left + right).capitalize() + "."
                    tape = normalize_letters(rendered)
                    if tape in KNOWN or tape in seen or len(tape) < MIN_LETTERS:
                        stats["known_or_duplicate_reject"] += 1
                        continue
                    seen.add(tape)
                    audit = _audit(rendered)
                    rows.append({
                        "rendered": rendered,
                        "frame": frame_name,
                        "dependency_state": frame["state"],
                        "center_letter": center,
                        "left_words": list(left),
                        "right_words": list(right),
                        "audit": audit,
                        "reader_status": "not_run; repair output is not human readability evidence",
                        "provenance": {
                            "source": "hand-authored alternatives inherited from the preflighted FSM",
                            "source_sentences_copied": False,
                            "known_catalogue_excluded": True,
                        },
                    })
                    stats["exact"] += int(audit["exact"])
                    stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
                    if len(rows) >= 120:
                        break
                if len(rows) >= 120:
                    break
            if len(rows) >= 120:
                break
        if len(probes) < 120:
            for left in itertools.islice(itertools.product(*left_options), 20):
                if _valid(left):
                    probes.append({
                        "frame": frame_name,
                        "left_words": list(left),
                        "left_tape": "".join(left),
                        "state": frame["state"],
                        "status": "complete-question-probe",
                    })
    rows.sort(key=lambda row: (-row["audit"]["mechanically_admitted"], -row["audit"]["letters"], row["rendered"]))
    admitted = [row for row in rows if row["audit"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_repair_no_reader_promotion",
        "method": "The same explicit question/answer dependency automaton is joined by a target-indexed slot-prefix walk: answer alternatives are considered only when they match the required center-plus-reverse tape prefix.",
        "repair_of": "interrogative-quantifier-fsm-20260915",
        "novelty_preflight": {
            "registry_entries_before_run": 90,
            "excluded_routes_before_run": 6,
            "status": "formal_preflight_before_execution",
            "signature_overlap": ["interrogative-quantifier-fsm-20260915"],
            "manual_review_required": True,
            "disposition": "concrete repair of the registered family; not counted as a new family",
        },
        "config": {
            "frame_count": len(FRAMES),
            "max_left_yields_per_frame": max_left,
            "center_letters": 26,
            "right_frame_materialized": False,
            "catalogue_text_imported": False,
        },
        "stats": {
            **dict(stats),
            "rendered_candidates": len(rows),
            "reader_eligible": 0,
            "mechanically_admitted": len(admitted),
        },
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": "same hand-authored alternatives as the registered FSM; only the join operator changed",
            "source_sentences_copied": False,
            "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"],
            "readability_certificate": False,
        },
        "next_repair": "Use a role-complete discourse grammar with independently authored lexical realizations; do not widen this slot inventory or rematerialize its Cartesian product.",
        "reader_gate": "No row is reader evidence. Any exact row must be manually screened and frozen with randomized intact/shuffled controls before readability claims.",
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
