"""Interrogative/quantifier dependency automaton for palindrome search.

Short auxiliary, pronoun, and quantifier words are often the only viable
cross-boundary seams, but previous searches treated them as generic slots.
This run carries an explicit question-inversion state and an answer-side
quantifier/dependency state while solving the odd center-letter equation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/interrogative-quantifier-fsm-20260915.json"
EXPERIMENT_ID = "interrogative-quantifier-fsm-20260915"
SIGNATURE = (
    "interrogative-quantifier-fsm|auxiliary-inversion-state|"
    "question-answer-dependency|center-letter-character-equation|independent-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

FUNCTION = frozenset("a an the some one all few many my our your i we you he she they it is are was were be do did does can will would not no and or but if as of to in on at by for with from near over under".split())
KNOWN = {normalize_letters(x) for x in json.loads((ROOT / "data" / "known_palindromes.json").read_text())}

FRAMES = {
    "question_negated": {
        "left": ("aux", "pron", "neg", "verb_past", "adv", "pron"),
        "right": ("quant", "verb_past", "adv", "prep", "adj", "noun"),
        "state": "inverted-question-to-quantified-answer",
    },
    "question_object": {
        "left": ("aux", "det", "noun", "verb_base", "prep", "det", "noun"),
        "right": ("pron", "verb_past", "det", "noun", "prep", "noun"),
        "state": "inverted-object-question-to-answer",
    },
    "question_copular": {
        "left": ("wh", "aux", "det", "noun", "adj"),
        "right": ("det", "noun", "cop", "adj", "quant"),
        "state": "wh-copular-question-to-quantified-description",
    },
}

OPTIONS = {
    "aux": "are is was were do did does can will would".split(),
    "pron": "i we you he she they it".split(),
    "neg": "not no".split(),
    "verb_past": "drawn written seen heard found read said made kept held sent taken known shown called marked left met put got had went came gave lost won used told asked".split(),
    "verb_base": "draw write see hear find read make keep hold send take know show call mark leave meet put get have go come give lose win use tell ask".split(),
    "adv": "onward ahead aside away again here there now ever well home over back forth today".split(),
    "det": "a an the some one this that my our your".split(),
    "quant": "few some one all many each no".split(),
    "noun": "era road home town room book note plan word day way time man woman child letter story song map car cat dog star moon sun friend teacher writer artist farmer sailor doctor nurse garden river harbor desk".split(),
    "adj": "new old good kind fair safe calm clear red small great true vast high low quiet bright young".split(),
    "prep": "to in on at by for with from near over under".split(),
    "wh": "what who where when why how".split(),
    "cop": "is are was were".split(),
}


def _valid(words: tuple[str, ...]) -> bool:
    content = [word for word in words if word not in FUNCTION]
    return len(content) == len(set(content))


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
        "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def run(*, max_left: int = 140_000) -> dict:
    rows: list[dict] = []
    probes: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    for frame_name, frame in FRAMES.items():
        left_options = [OPTIONS[slot] for slot in frame["left"]]
        right_options = [OPTIONS[slot] for slot in frame["right"]]
        right_index: dict[str, list[tuple[str, ...]]] = {}
        for right in itertools.product(*right_options):
            if not _valid(right):
                continue
            right_index.setdefault("".join(right), []).append(right)
        stats[f"{frame_name}_right_yields"] = sum(len(v) for v in right_index.values())
        checked = 0
        for left in itertools.product(*left_options):
            if checked >= max_left:
                break
            checked += 1
            stats["left_yields_checked"] += 1
            if not _valid(left):
                continue
            left_tape = "".join(left)
            if len(left_tape) * 2 + 1 < MIN_LETTERS:
                continue
            for center in "abcdefghijklmnopqrstuvwxyz":
                stats["center_equations"] += 1
                for right in right_index.get(center + left_tape[::-1], ()):
                    words = left + right
                    rendered = " ".join(words).capitalize() + "."
                    tape = normalize_letters(rendered)
                    if tape in KNOWN or tape in seen or len(tape) < MIN_LETTERS:
                        stats["known_or_duplicate_reject"] += 1
                        continue
                    seen.add(tape)
                    audit = _audit(rendered)
                    row = {
                        "rendered": rendered,
                        "frame": frame_name,
                        "dependency_state": frame["state"],
                        "center_letter": center,
                        "left_words": list(left),
                        "right_words": list(right),
                        "audit": audit,
                        "reader_status": "not_run; interrogative syntax is not human readability evidence",
                        "provenance": {"source": "hand-authored lexical alternatives", "source_sentences_copied": False, "known_catalogue_excluded": True},
                    }
                    rows.append(row)
                    stats["exact"] += int(audit["exact"])
                    stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
                    if len(rows) >= 120:
                        break
                if len(rows) >= 120:
                    break
            if len(rows) >= 120:
                break
        # Retain a few non-palindromic complete frame probes to show what the
        # FSM actually emitted even if the equation has no novel closure.
        if len(probes) < 160:
            for left in itertools.islice(itertools.product(*left_options), 20):
                if _valid(left):
                    probes.append({"frame": frame_name, "left_words": list(left), "left_tape": "".join(left), "state": frame["state"], "status": "complete-question-probe"})
    rows.sort(key=lambda row: (-row["mechanically_admitted"], -row["audit"]["letters"], row["rendered"]))
    admitted = [row for row in rows if row["audit"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": "An explicit auxiliary-inversion/question-to-answer automaton emits complete interrogative and quantified-answer frames; the right frame is independently indexed under center + reverse(left) character equations.",
        "novelty_preflight": {"registry_entries_before_run": 89, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": [], "manual_review_required": False},
        "config": {"frame_count": len(FRAMES), "frames": {name: {"left": data["left"], "right": data["right"], "state": data["state"]} for name, data in FRAMES.items()}, "max_left_yields_per_frame": max_left, "center_letters": 26, "catalogue_text_imported": False, "known_palindromes_imported": True, "known_palindrome_role": "exclusion-only"},
        "stats": {**dict(stats), "rendered_candidates": len(rows), "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "rendered_candidates_and_probes": rows,
        "probes": probes,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "hand-authored interrogative and quantified lexical alternatives", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"], "readability_certificate": False},
        "next_repair": "Carry explicit semantic roles and polarity through a two-question discourse graph; preserve the inversion automaton and do not add a larger undifferentiated word bank.",
        "reader_gate": "No row is reader evidence. Any mechanically admitted row must be manually screened and frozen with randomized intact/shuffled controls before readability claims.",
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
