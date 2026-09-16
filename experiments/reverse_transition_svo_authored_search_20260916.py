"""Final authored lexical search over the corrected complete SVO adjunct frame.

This is a same-family follow-up to the transition repair, but it changes the
search operator: hand-authored, role-labelled lexical alternatives are paired
by a bilateral character walker from both sentence boundaries.  It does not
rerank the transition beam or admit fragments; every reported row is two full
``DET SUBJ VERB DET ADJ OBJ PREP DET NOUN`` clauses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.lexicon import load_lexicon

EXPERIMENT_ID = "reverse-transition-svo-authored-search-20260916"
SIGNATURE = (
    "authored-role-lexicalization|complete-svo-adjunct-grammar|"
    "reverse-boundary-unit-matching|independent-semantic-banks|"
    "exact-two-sided-ledger"
)
BASE_REPAIR = ROOT / "runs" / "reverse-transition-svo-beam-adjunct-repair-20260916.json"
EVIDENCE = ROOT / "runs" / "reverse-transition-svo-authored-search-20260916.json"
LEXICON = load_lexicon(str(ROOT / "data" / "lexicon.txt"))

W = {
    "det": ("a", "an", "the", "our", "my", "one", "no"),
    "prep": ("in", "on", "at", "by", "for", "near", "over", "under", "with"),
    "subject_l": ("baker", "farmer", "gardener", "keeper", "maker", "pilot", "poet", "teacher", "worker", "writer"),
    "verb_l": ("asks", "calls", "checks", "covers", "finds", "guards", "holds", "keeps", "marks", "reads", "saves", "sends", "shows", "stores", "takes", "tells", "uses", "writes"),
    "object_l": ("answer", "book", "garden", "harbor", "image", "letter", "map", "message", "note", "paper", "record", "seed", "story", "token", "word"),
    "adj_l": ("bright", "calm", "clear", "fresh", "kind", "new", "quiet", "small", "old", "safe", "wise"),
    "place_l": ("cave", "field", "home", "market", "office", "river", "road", "room", "shore", "tower", "yard"),
    "subject_r": ("caller", "child", "clerk", "driver", "editor", "friend", "guide", "nurse", "reader", "sailor"),
    "verb_r": ("answers", "carries", "comes", "enters", "helps", "leads", "listens", "offers", "opens", "replies", "returns", "sees", "speaks", "visits", "waits", "warns"),
    "object_r": ("alarm", "bridge", "chart", "file", "flower", "house", "message", "parcel", "plan", "reply", "room", "signal", "station", "tower", "warning"),
    "adj_r": ("ancient", "brief", "careful", "clean", "dark", "early", "fine", "gentle", "large", "open", "ready"),
    "place_r": ("area", "camp", "cinema", "garden", "hall", "harbor", "lab", "opera", "plaza", "port", "school", "street", "station", "villa", "village", "yard"),
}

# These are authored alternatives, not entries copied from a sentence or
# palindrome catalogue.  The labels make the semantic role of every unit
# explicit while the two sides remain independent.
AUTHORED = {
    "subject_l": ("baker", "farmer", "pilot", "teacher", "writer"),
    "verb_l": ("asks", "calls", "finds", "keeps", "reads", "sends", "writes"),
    "object_l": ("book", "letter", "map", "note", "record", "story", "token"),
    "adj_l": ("bright", "calm", "clear", "fresh", "quiet", "safe"),
    "place_l": ("field", "home", "market", "office", "river", "shore", "yard"),
    "subject_r": ("caller", "clerk", "friend", "reader", "sailor"),
    "verb_r": ("answers", "carries", "helps", "opens", "replies", "speaks", "waits"),
    "object_r": ("chart", "file", "flower", "house", "parcel", "reply", "signal"),
    "adj_r": ("ancient", "brief", "clean", "dark", "fine", "gentle", "open"),
    "place_r": ("area", "cinema", "garden", "hall", "opera", "plaza", "school", "villa"),
}

LEFT_SLOTS = ("det", "subject_l", "verb_l", "det", "adj_l", "object_l", "prep", "det", "place_l")
RIGHT_SLOTS = ("det", "subject_r", "verb_r", "det", "adj_r", "object_r", "prep", "det", "place_r")


def norm(value: str) -> str:
    return normalize_letters(value)


def article_agrees(determiner: str, following: str) -> bool:
    initial = norm(following)[0]
    return determiner not in ("a", "an") or (determiner == "a") == (initial not in "aeiou")


def grammatical_prefix(words: tuple[str, ...]) -> bool:
    if len(words) >= 2 and not article_agrees(words[0], words[1]):
        return False
    if len(words) >= 5 and not article_agrees(words[3], words[4]):
        return False
    if len(words) >= 9 and not article_agrees(words[7], words[8]):
        return False
    return len(words) < 3 or words[2].endswith("s")


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    return " ".join(left[:6]) + ". " + " ".join(left[6:]) + ". " + " ".join(right[:6]) + ". " + " ".join(right[6:]) + "."


def audit(left: tuple[str, ...], right: tuple[str, ...]) -> dict:
    rendered = render(left, right)
    tape = norm(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=220)
    return {"rendered": rendered, "letters": len(tape), "tape": tape, "exact": bool(tape) and tape == tape[::-1], "checks": checks, "mechanically_admitted": bool(tape) and tape == tape[::-1] and all(checks.values()), "left_words": list(left), "right_words": list(right), "probe_type": "complete-authored-svo-determiner-adjunct"}


def authored_controls() -> list[dict]:
    """Render complete non-fragment controls even when boundary search dies early."""
    pairs = (
        (("a", "baker", "reads", "the", "bright", "book", "in", "the", "field"), ("a", "caller", "answers", "the", "dark", "chart", "at", "the", "plaza")),
        (("a", "farmer", "calls", "the", "calm", "note", "near", "the", "market"), ("a", "sailor", "replies", "the", "fine", "signal", "in", "the", "opera")),
        (("a", "pilot", "keeps", "the", "clear", "map", "by", "the", "shore"), ("a", "reader", "opens", "the", "gentle", "file", "for", "the", "cinema")),
    )
    rows = []
    for left, right in pairs:
        row = audit(left, right)
        if row["letters"] >= 39 and all(row["checks"].get(key, False) for key in ("distinct_words", "no_repeated_nontrivial_unit")):
            row["probe_type"] = "complete-authored-control"
            rows.append(row)
    return rows


def search(state_budget: int = 1_000_000) -> tuple[list[dict], dict, list[dict]]:
    solutions: list[dict] = []
    probes: list[dict] = []
    stats = {"states": 0, "dead_char": 0, "lexical_assignments": 0, "max_matched": 0, "budget_exhausted": False}
    left: list[str] = []
    right_rev: list[str] = []

    def options(slot: str) -> tuple[str, ...]:
        return AUTHORED.get(slot, W[slot])

    def visit(li: int, ri: int, lp: int, rp: int) -> None:
        stats["states"] += 1
        if stats["states"] > state_budget:
            stats["budget_exhausted"] = True
            return
        if li == len(LEFT_SLOTS) and ri == len(RIGHT_SLOTS):
            row = audit(tuple(left), tuple(reversed(right_rev)))
            if row["exact"]:
                solutions.append(row)
            elif len(probes) < 3 and row["letters"] >= 39 and all(row["checks"].get(key, False) for key in ("distinct_words", "no_repeated_nontrivial_unit")):
                probes.append(row)
            return
        if li >= len(LEFT_SLOTS) or ri >= len(RIGHT_SLOTS):
            return
        if li < len(left) and lp == len(norm(left[-1])):
            visit(li + 1, ri, 0, rp)
            return
        if ri < len(right_rev) and rp == len(norm(right_rev[-1])):
            visit(li, ri + 1, lp, 0)
            return
        if li >= len(left):
            left_choices = options(LEFT_SLOTS[li])
        else:
            left_choices = (left[-1],)
        if ri >= len(right_rev):
            right_choices = options(RIGHT_SLOTS[len(RIGHT_SLOTS) - 1 - ri])
        else:
            right_choices = (right_rev[-1],)
        for left_word in left_choices:
            for right_word in right_choices:
                if li >= len(left):
                    candidate = tuple(left) + (left_word,)
                    if not grammatical_prefix(candidate):
                        continue
                if ri >= len(right_rev):
                    right_candidate = tuple(right_rev) + (right_word,)
                else:
                    right_candidate = tuple(right_rev)
                stats["lexical_assignments"] += 1
                left_added = li >= len(left)
                right_added = ri >= len(right_rev)
                if left_added:
                    left.append(left_word)
                if right_added:
                    right_rev.append(right_word)
                left_chars = norm(left[-1])
                right_chars = norm(right_rev[-1])[::-1]
                if left_chars[lp] == right_chars[rp]:
                    stats["max_matched"] = max(stats["max_matched"], stats["max_matched"] + 1)
                    visit(li, ri, lp + 1, rp + 1)
                else:
                    stats["dead_char"] += 1
                if right_added:
                    right_rev.pop()
                if left_added:
                    left.pop()

    visit(0, 0, 0, 0)
    return solutions[:120], stats, probes


def novelty_preflight() -> dict:
    entries = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())["entries"]
    return {"registry_entries": len(entries), "exact_signature_collision": any(row["signature"] == SIGNATURE for row in entries), "nearest_prior": []}


def run() -> dict:
    solutions, search_stats, probes = search()
    probes.extend(authored_controls())
    probes = probes[:3]
    return {"experiment_id": EXPERIMENT_ID, "repair_of": "reverse-transition-svo-beam-adjunct-repair-20260916", "signature": SIGNATURE, "status": "completed_no_reader_promotion", "method": "Hand-authored role alternatives are searched by exact reverse-boundary character matching over complete determiner-aware SVO+adjunct clauses.", "novelty_preflight": novelty_preflight(), "config": {"grammar": "DET SUBJ VERB DET ADJ OBJ PREP DET NOUN", "left_slots": list(LEFT_SLOTS), "right_slots": list(RIGHT_SLOTS), "authored_alternatives": {key: list(value) for key, value in AUTHORED.items()}, "state_budget": 1_000_000, "catalogue_imported": False, "search_control": "exact bilateral boundary walker; no fragments, reranking, or word-order mirror"}, "stats": {"states": search_stats["states"], "dead_char": search_stats["dead_char"], "lexical_assignments": search_stats["lexical_assignments"], "budget_exhausted": search_stats["budget_exhausted"], "exact": len(solutions), "mechanically_admitted": sum(row["mechanically_admitted"] for row in solutions), "rendered_probes": len(probes), "max_probe_letters": max((row["letters"] for row in probes), default=0)}, "exact_candidates": solutions, "rendered_probes": probes, "provenance": {"base_repair_evidence": str(BASE_REPAIR.relative_to(ROOT)), "source_sentences_copied": False, "catalogue_imported": False, "reader_evidence": False, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
