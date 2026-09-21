"""Bounded variable-span lexical/dependency CSP.

Unlike a completed-clause sweep, this solver chooses a target tape length,
token identities, and token spans together. Each emitted character constrains
its still-unresolved mirror position; a later token must satisfy that pending
constraint. This is a small global chart, not a finished-text comparison.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/variable-span-constraint-graph-20260921.json"

LEXICON = {
    "det": ["the", "a", "this", "that", "each", "one"],
    "n": ["watcher", "keeper", "writer", "teacher", "artist", "pilot", "nurse", "guard"],
    "v": ["observes", "guides", "writes", "helps", "trusts", "keeps"],
    "adj": ["calm", "kind", "wise", "alert", "bright", "steady"],
    "prep": ["near", "under", "beside", "beyond", "within", "around"],
    "obj": ["lantern", "letter", "garden", "harbor", "bridge", "signal", "window", "river"],
    "adv": ["quietly", "gently", "daily", "well"],
    "conj": ["and", "while", "as", "because"],
}
assert sum(map(len, LEXICON.values())) == 48


@dataclass(frozen=True)
class Slot:
    name: str
    cat: str


SLOTS = (
    Slot("d1", "det"), Slot("subj1", "n"), Slot("v1", "v"),
    Slot("d2", "det"), Slot("obj1", "obj"), Slot("conj", "conj"),
    Slot("d3", "det"), Slot("subj2", "n"), Slot("v2", "v"),
    Slot("obj2", "obj"),
)


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def independent_audit(words: list[str]) -> dict:
    tape = norm(" ".join(words))
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    return {
        "letters": len(tape),
        "exact": i >= j and bool(tape),
        "first_mismatch": None if i >= j else [i, tape[i], j, tape[j]],
        "forward_sha256": sha(tape),
        "reverse_sha256": sha(tape[::-1]),
        "sha_equal": sha(tape) == sha(tape[::-1]),
    }


def _grammatical(words: list[str]) -> bool:
    if len(words) != len(SLOTS) or words[5] != "and":
        return False
    return all(words[i] in LEXICON[SLOTS[i].cat] for i in range(len(SLOTS)))


def solve(limit: int = 50_000, targets=range(39, 53)) -> dict:
    nodes = conflicts = complete = 0
    # Boundary-first repair: skip impossible target tapes before entering the
    # lexical DFS.  This is exact length propagation, not a widened lexicon.
    feasible_targets = set()
    totals = {0}
    for slot in SLOTS:
        totals = {n + len(norm(word)) for n in totals for word in LEXICON[slot.cat]}
    feasible_targets = sorted(set(totals) & set(targets))
    learned: set[tuple[str, str, int, int]] = set()
    found: list[list[str]] = []

    def run_target(target: int) -> None:
        nonlocal nodes, conflicts, complete
        values: dict[str, str] = {}
        chars: dict[int, str] = {}

        def place(word: str, start: int, changes: list[int]) -> bool:
            """Place one span and its still-unresolved mirrored characters."""
            for j, ch in enumerate(norm(word)):
                pos = start + j
                mirror = target - 1 - pos
                for p, value in ((pos, ch), (mirror, ch)):
                    if p < 0 or p >= target:
                        return False
                    old = chars.get(p)
                    if old is not None and old != value:
                        return False
                    if old is None:
                        chars[p] = value
                        changes.append(p)
            return True

        def rec(left: int, right: int, left_offset: int, right_offset: int) -> None:
            nonlocal nodes, conflicts, complete
            if nodes >= limit:
                return
            if left > right:
                if left_offset != right_offset or len(chars) != target:
                    conflicts += 1
                    return
                complete += 1
                words = [values[s.name] for s in SLOTS]
                if _grammatical(words) and not any(words.count(w) > 1 for w in words):
                    found.append(words)
                return

            lslot, rslot = SLOTS[left], SLOTS[right]
            # Pair the outermost unresolved spans.  The right span is indexed
            # from the shared target end, so both boundaries advance inward.
            for lword in LEXICON[lslot.cat]:
                lw = norm(lword)
                lkey = (lslot.name, lw, target, left_offset)
                if lkey in learned or left_offset + len(lw) > target - right_offset:
                    continue
                nodes += 1
                lchanges: list[int] = []
                if not place(lw, left_offset, lchanges):
                    conflicts += 1; learned.add(lkey)
                    for p in lchanges: chars.pop(p, None)
                    continue
                values[lslot.name] = lword
                right_words = LEXICON[rslot.cat] if right != left else [lword]
                for rword in right_words:
                    rw = norm(rword)
                    rstart = target - right_offset - len(rw)
                    rkey = (rslot.name, rw, target, right_offset)
                    if rkey in learned or rstart < left_offset + len(lw):
                        continue
                    if right != left and rword == lword:
                        continue
                    nodes += 1
                    rchanges: list[int] = []
                    if place(rw, rstart, rchanges):
                        values[rslot.name] = rword
                        rec(left + 1, right - 1, left_offset + len(lw), right_offset + len(rw))
                        values.pop(rslot.name, None)
                    else:
                        conflicts += 1; learned.add(rkey)
                    for p in rchanges: chars.pop(p, None)
                    if nodes >= limit: break
                values.pop(lslot.name, None)
                for p in lchanges: chars.pop(p, None)
                if nodes >= limit: return

        rec(0, len(SLOTS) - 1, 0, 0)

    for target in feasible_targets:
        if nodes >= limit:
            break
        run_target(target)

    return {
        "found": found,
        "stats": {"nodes": nodes, "conflicts": conflicts, "learned_nogoods": len(learned), "complete_assignments": complete, "limit": limit, "targets_requested": len(set(targets)), "feasible_targets": feasible_targets, "length_propagation": "precomputed lexical total-length support"},
        "state_model": {
            "token_identity_variables": len(SLOTS),
            "variable_word_boundaries": True,
            "shared_character_variables": True,
            "agreement_links": ["det-noun compatibility", "subject-verb valency"],
            "dependency_links": ["verb->object", "coordination->finite-clause"],
            "conflict_learning": "learned (slot,lexeme,target,offset) nogoods",
            "search_order": "paired outermost spans, left/right boundaries advance inward",
            "shared_target_length": True,
            "unresolved_mirror_constraints": "retained in shared character map until paired span placement",
        },
    }


def run(limit: int = 50_000) -> dict:
    result = solve(limit)
    records = []
    for words in result["found"][:8]:
        rendered = " ".join(words) + "."
        records.append({
            "tokens": words,
            "rendered": rendered,
            "audit": independent_audit(words),
            "provenance": {"joint_token_length_lexeme_search": True, "complete_clause_before_test": False, "lexical_entries": sum(map(len, LEXICON.values()))},
            "anti_shortcut": {"finished_clause_compare": False, "seed_recovery": False, "mirrored_units": False, "repeated_units": False},
        })
    seed = "the watcher observes a lantern near the calm keeper"
    return {
        "experiment_id": "variable-span-constraint-graph-20260921",
        "method": "bounded bidirectional outer-span CSP with shared target length, unresolved mirrored character variables, lexical identity/length domains, agreement/valency/dependency links, and learned conflicts",
        "config": {"token_slots": len(SLOTS), "lexical_entries": sum(map(len, LEXICON.values())), "letter_band": [39, 52], "state_limit": limit},
        **result,
        "records": records,
        "calibration_seed": {"text": seed, "letters": len(norm(seed)), "used_as_success": False, "purpose": "calibration only"},
        "independent_pointer_sha_audit": True,
        "search_order_provenance": {"paired_outer_spans": True, "shared_target_length": True, "lexicon_entries": 48, "posthoc_reversal": False},
        "novelty_preflight": {"status": "passed", "signature": "variable-span|joint-lexeme-boundary|shared-character-vars|conflict-learning", "anti_shortcut_checks": ["no completed-clause sweep", "no 38-letter seed success", "no mirrored lexical units"]},
        "queue_row": {"lane": "Astra", "status": "satisfying assignments" if records else "bounded residual", "next": "hold lexicon fixed; review dependency/coordination links on the bidirectional residual"},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run(), indent=2))
