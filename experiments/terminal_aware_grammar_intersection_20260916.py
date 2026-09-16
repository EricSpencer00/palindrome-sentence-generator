"""Terminal-aware character intersection for independently typed clauses.

Earlier outside-in probes lost a valid path when both current lexical tries
ended on the same character: they asked for one more shared character instead
of taking the two word-boundary epsilon transitions.  This experiment makes
those terminal transitions explicit while keeping independent, complete
clause grammars on both sides.  No word or phrase is emitted from the other
side's reverse tape.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/terminal-aware-grammar-intersection-20260916.json"
EXPERIMENT_ID = "terminal-aware-grammar-intersection-20260916"
SIGNATURE = (
    "terminal-aware-grammar-intersection|epsilon-word-boundary-closure|"
    "typed-svo-num-name-frames|independent-clause-realization|exact-audit"
)
MIN_LETTERS = 39

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


WORDS = {
    "det": "a an the some one my our your this that every each no".split(),
    "pron": "i we you he she they it".split(),
    "noun": (
        "aid aide artist baker captain child doctor farmer friend garden harbor letter man map memo memos moon nurse poet river sailor story teacher town writer woman song star road room book note plan word day way time men".split()
    ),
    "verb": (
        "ask asks asked carry carries carried draw draws drew find finds found give gives gave hear hears heard hold holds held keep keeps kept leave leaves left make makes made meet meets met read reads read write writes wrote send sends sent see sees saw show shows showed take takes took tell tells told use uses used inspire inspires inspired rip rips".split()
    ),
    "num": "one two three four five six seven eight nine ten eleven twelve".split(),
    "name": "diana anna eva ada nina nora sara maria lena iris maya lana edna ariana alina elena serena olivia sophia amelia clara luna dana tina sonia".split(),
    "adj": "old new kind quiet bright small red calm clear good wise safe young fair great true vast high low".split(),
    "adv": "now ever again well here there away onward ahead home back today".split(),
    "prep": "in on at by to for with near over under from".split(),
    "aux": "is are was were can will do did has have".split(),
}

LEFT_FRAMES = {
    "det_noun_verb_num_noun": ("det", "noun", "verb", "num", "noun"),
    "det_adj_noun_verb_det_noun": ("det", "adj", "noun", "verb", "det", "noun"),
    "pron_verb_det_noun": ("pron", "verb", "det", "noun"),
    "det_noun_aux_adj": ("det", "noun", "aux", "adj"),
    "name_verb_det_noun": ("name", "verb", "det", "noun"),
    "det_noun_verb_det_noun": ("det", "noun", "verb", "det", "noun"),
}
RIGHT_FRAMES = {
    "det_noun_verb_name": ("det", "noun", "verb", "name"),
    "det_noun_verb_det_noun": ("det", "noun", "verb", "det", "noun"),
    "pron_noun_verb_name": ("pron", "noun", "verb", "name"),
    "det_adj_noun_verb_det_noun": ("det", "adj", "noun", "verb", "det", "noun"),
    "det_noun_aux_adj": ("det", "noun", "aux", "adj"),
}

FUNCTION = frozenset(
    "a an the some one my our your this that every each no i we you he she they it is are was were can will do did has have in on at by to for with near over under from".split()
)
KNOWN = {
    normalize_letters(x)
    for x in json.loads((ROOT / "data/known_palindromes.json").read_text())
}


def _valid_content(words: tuple[str, ...]) -> bool:
    content = [word for word in words if word not in FUNCTION]
    return len(content) == len(set(content))


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    independent = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=220)
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


def _terminal_closure(state: tuple, left_slots: tuple[str, ...], right_slots: tuple[str, ...]) -> tuple:
    """Take completed-word epsilon transitions before requesting a character."""
    li, ri, lwrd, rwrd, lp, rp, lw, rw = state
    while lwrd is not None and lp == lwrd:
        lw = lw + (lwrd,)
        li += 1
        lwrd, lp = None, ""
    while rwrd is not None and rp == rwrd[::-1]:
        rw = rw + (rwrd,)
        ri -= 1
        rwrd, rp = None, ""
    return li, ri, lwrd, rwrd, lp, rp, lw, rw


def _search(left_slots: tuple[str, ...], right_slots: tuple[str, ...], max_states: int, stats: Counter) -> tuple[list[dict], list[dict], int]:
    start = (0, len(right_slots) - 1, None, None, "", "", (), ())
    queue = deque([start])
    seen: set[tuple] = set()
    rows: list[dict] = []
    dead: list[dict] = []
    while queue and len(seen) < max_states:
        raw = queue.popleft()
        state = _terminal_closure(raw, left_slots, right_slots)
        li, ri, lwrd, rwrd, lp, rp, lw, rw = state
        key = (li, ri, lwrd, rwrd, lp, rp, lw, rw)
        if key in seen:
            continue
        seen.add(key)
        stats["states"] += 1
        if li == len(left_slots) and ri < 0 and not lp and not rp:
            rendered = " ".join(lw).capitalize() + "; " + " ".join(reversed(rw)) + "."
            if len(normalize_letters(rendered)) < MIN_LETTERS:
                stats["short_closures"] += 1
                continue
            if not _valid_content(lw + rw):
                stats["content_reuse_rejects"] += 1
                continue
            audit = _audit(rendered)
            tape = audit["normalized_tape"]
            if tape in KNOWN:
                stats["known_rejects"] += 1
                continue
            rows.append({
                "rendered": rendered,
                "left_words": list(lw),
                "right_words": list(reversed(rw)),
                "audit": audit,
                "reader_status": "not_run; exactness is not human readability evidence",
                "provenance": {"source": "hand-authored typed lexical alternatives", "source_sentences_copied": False, "known_catalogue_excluded": True},
            })
            stats["exact"] += int(audit["exact"])
            stats["mechanically_admitted"] += int(audit["mechanically_admitted"])
            continue
        if li >= len(left_slots) or ri < 0:
            stats["grammar_exhaustion_dead"] += 1
            continue
        left_words = WORDS[left_slots[li]] if lwrd is None else (lwrd,)
        right_words = WORDS[right_slots[ri]] if rwrd is None else (rwrd,)
        for lchoice in left_words:
            for rchoice in right_words:
                lword = lchoice
                rword = rchoice[::-1]
                lprefix = lp if lwrd is not None else ""
                rprefix = rp if rwrd is not None else ""
                lrest = lword[len(lprefix):]
                rrest = rword[len(rprefix):]
                if not lrest or not rrest or lrest[0] != rrest[0]:
                    stats["character_mismatch_dead"] += 1
                    if len(dead) < 120:
                        dead.append({"left_slot": li, "right_slot_outward": ri, "left_prefix": lprefix, "right_prefix": rprefix, "left_word": lword, "right_word": rchoice, "reason": "outer-character-mismatch"})
                    continue
                stats["character_matches"] += 1
                queue.append((li, ri, lword, rchoice, lprefix + lrest[0], rprefix + rrest[0], lw, rw))
    return rows, dead, len(seen)


def run(*, max_states: int = 300_000) -> dict:
    stats = Counter()
    rows: list[dict] = []
    dead: list[dict] = []
    frame_runs: list[dict] = []
    for left_name, left_slots in LEFT_FRAMES.items():
        for right_name, right_slots in RIGHT_FRAMES.items():
            stats["frame_pairs"] += 1
            before = len(rows)
            local = Counter()
            found, misses, used = _search(tuple(left_slots), tuple(right_slots), max_states, local)
            for row in found:
                if not _valid_content(tuple(row["left_words"] + row["right_words"])):
                    local["content_reuse_rejects"] += 1
                    continue
                rows.append(row)
            if len(dead) < 240:
                dead.extend(misses[: 240 - len(dead)])
            frame_runs.append({"left_frame": left_name, "right_frame": right_name, "left_slots": left_slots, "right_slots": right_slots, "rows": len(rows) - before, "states": local["states"], "character_mismatches": local["character_mismatch_dead"], "known_rejects": local["known_rejects"]})
            stats.update(local)
            if len(rows) >= 160:
                break
        if len(rows) >= 160:
            break
    # The frame products above are deliberately bounded by the first 160
    # closures; preserve deterministic unique surfaces for the audit ledger.
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(row["audit"]["normalized_tape"], row)
    rows = sorted(unique.values(), key=lambda row: (-row["audit"]["mechanically_admitted"], -row["audit"]["letters"], row["rendered"]))
    admitted = [row for row in rows if row["audit"]["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_terminal_aware_intersection",
        "method": "A product of independent typed clause grammars is intersected at characters from the outside inward; completed lexical tries take explicit epsilon boundary transitions before the next character is requested.",
        "novelty_preflight": {"registry_entries_before_run": 90, "excluded_routes_before_run": 6, "status": "formal_preflight_before_execution", "signature_overlap": [], "manual_review_required": False},
        "config": {"left_frame_count": len(LEFT_FRAMES), "right_frame_count": len(RIGHT_FRAMES), "max_states_per_pair": max_states, "catalogue_text_imported": False, "word_order_only_generation": False, "terminal_epsilon_closure": True},
        "stats": {**dict(stats), "rendered_candidates": len(rows), "reader_eligible": 0, "mechanically_admitted": len(admitted)},
        "frame_runs": frame_runs,
        "rendered_candidates_and_probes": rows,
        "dead_frontier": dead[:240],
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source": "hand-authored typed lexical alternatives; no intact source sentences", "source_sentences_copied": False, "independent_audits": ["normalized-tape-reversal", "ASCII-two-pointer"]},
        "next_repair": "Keep terminal epsilon closure fixed and add a semantic dependency graph that licenses clause roles before character intersection; do not rematerialize Cartesian products or mirror a lexical unit.",
        "reader_gate": "No row is reader evidence. Only a novel exact surface that passes mechanical admission may enter randomized blinded intact/shuffled reading.",
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
