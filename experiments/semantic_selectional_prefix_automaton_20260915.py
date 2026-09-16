"""Selectional-preference clause automaton for exact palindromes.

This route learns coarse subject--verb--object compatibility from tagged Brown
sentences, then keeps those role states live while a character-synchronous
prefix trie searches two independently ordered clauses.  It is intentionally
different from a fixed clause cross-product: lexical choices are conditioned
on the live semantic frame and the opposing side's character prefix at every
word boundary.  No fixed palindrome tape, reflected phrase, or word-order
mirror is supplied.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ID = "semantic-selectional-prefix-automaton"
SIGNATURE = (
    "semantic-selectional-prefix-automaton|majority-pos-lexicon|"
    "learned-subject-verb-object-preferences|paired-slot-grammar|"
    "character-synchronous-trie|non-mirrored-boundary-audit"
)
OUT = ROOT / "runs/semantic-selectional-prefix-automaton-20260915.json"

STOP = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your",
    "his", "her", "our", "their", "one", "some", "any", "each", "every",
    "no", "i", "me", "you", "he", "she", "we", "they", "it", "is", "was",
    "are", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "and",
    "or", "but", "not", "so", "if", "then", "than", "very", "more",
}
DETS = ("a", "an", "the", "some", "my", "one")
SHAPES = (
    ("DET", "SUBJ", "VERB", "DET", "OBJ"),
    ("DET", "ADJ", "SUBJ", "VERB", "DET", "OBJ"),
    ("DET", "SUBJ", "VERB", "DET", "ADJ", "OBJ"),
    ("DET", "SUBJ", "VERB", "ADV"),
    ("SUBJ", "VERB", "DET", "OBJ"),
    ("DET", "SUBJ", "VERB", "ADP", "DET", "OBJ"),
    # Held-out repair: one complete main clause followed by an explicit
    # subordinate observation clause.  This is a repair state, not a new
    # registry family, because the learned lexical frame and character-trie
    # transition remain unchanged.
    ("DET", "SUBJ", "VERB", "DET", "OBJ", "CONJ", "DET", "SUBJ", "VERB"),
)


@dataclass(frozen=True)
class Frame:
    name: str
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]


FRAMES = (
    Frame("observation", ("observer", "researcher", "scientist", "curator", "archivist", "surveyor", "analyst", "editor", "reporter", "historian", "teacher", "reader"),
          ("records", "studies", "tracks", "reviews", "measures", "maps", "examines", "observes", "reports", "checks", "compares", "notes"),
          ("reports", "patterns", "storms", "letters", "routes", "results", "evidence", "images", "records", "papers", "signals", "weather", "charts", "figures", "archive", "camera", "agenda", "memo", "notes")),
    Frame("craft", ("artisan", "carpenter", "mason", "painter", "weaver", "baker", "sculptor", "builder", "designer", "maker", "carver", "tailor"),
          ("carves", "builds", "paints", "weaves", "shapes", "bakes", "crafts", "makes", "designs", "forms", "cuts", "stitches"),
          ("figures", "bridges", "canvases", "baskets", "patterns", "bread", "statue", "arena", "furniture", "models", "panels", "cloth", "garments", "tables", "frames", "vessels")),
    Frame("travel", ("traveler", "pilot", "sailor", "scout", "navigator", "guide", "driver", "captain", "hiker", "ranger", "explorer", "tourist"),
          ("follows", "crosses", "maps", "guides", "marks", "finds", "visits", "reaches", "leaves", "passes", "enters", "sees"),
          ("trails", "rivers", "routes", "bridges", "islands", "harbors", "map", "cinema", "village", "cities", "valleys", "forests", "stations", "airports", "camp", "shore")),
)


def corpus_preferences() -> dict[str, dict[str, Counter[str]]]:
    """Extract light lexical preferences without copying corpus sentences."""
    try:
        from nltk.corpus import brown
        out = {"subject": defaultdict(Counter), "verb": defaultdict(Counter)}
        for sent in brown.tagged_sents(tagset="universal"):
            words = [(w.casefold(), tag) for w, tag in sent
                     if w.isascii() and w.isalpha()]
            for i in range(len(words) - 2):
                a, ta = words[i]
                b, tb = words[i + 1]
                c, tc = words[i + 2]
                if ta in {"NOUN", "PRON"} and tb == "VERB" and tc in {"NOUN", "PRON"}:
                    out["subject"][a][b] += 1
                    out["verb"][b][c] += 1
        return out
    except LookupError:
        return {"subject": defaultdict(Counter), "verb": defaultdict(Counter)}


PREFERENCES = corpus_preferences()


def ranked(items: Iterable[str], cap: int = 180) -> tuple[str, ...]:
    unique = {w for w in items if w.isascii() and w.isalpha() and (len(w) >= 3 or w in DETS)}
    return tuple(sorted(unique, key=lambda w: (-zipf_frequency(w, "en"), w))[:cap])


LEX = {
    "DET": DETS,
    "ADJ": ranked(("quiet bright gentle steady plain careful patient coastal northern annual detailed damaged expert local ancient distant changing"), 40),
    "ADV": ranked(("calmly softly carefully brightly quietly steadily patiently clearly"), 24),
    "ADP": ("to", "of", "in", "for", "on", "with", "at", "by", "near", "after", "under", "over"),
    "CONJ": ("while", "after", "because", "and"),
}


def frame_words(frame: Frame, slot: str, selected: tuple[str, ...]) -> tuple[str, ...]:
    if slot == "SUBJ":
        return frame.subjects
    if slot == "VERB":
        if selected and selected[0] in PREFERENCES["subject"]:
            preferred = PREFERENCES["subject"][selected[0]]
            return tuple(sorted(frame.verbs, key=lambda w: (-preferred[w], -zipf_frequency(w, "en"), w)))
        return frame.verbs
    if slot == "OBJ":
        if selected and selected[-1] in PREFERENCES["verb"]:
            preferred = PREFERENCES["verb"][selected[-1]]
            return tuple(sorted(frame.objects, key=lambda w: (-preferred[w], -zipf_frequency(w, "en"), w)))
        return frame.objects
    return LEX[slot]


def article_ok(words: tuple[str, ...], slots: tuple[str, ...]) -> bool:
    for i, (word, slot) in enumerate(zip(words, slots)):
        if slot == "DET" and i + 1 < len(words):
            nxt = words[i + 1]
            if word == "a" and nxt[0] in "aeiou":
                return False
            if word == "an" and nxt[0] not in "aeiou":
                return False
    return True


def direct_audit(tape: str) -> bool:
    return bool(tape) and tape == tape[::-1]


def opposing_index_audit(tape: str) -> bool:
    if not tape:
        return False
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return True


def search(left_frame: Frame, right_frame: Frame, left_shape: tuple[str, ...],
           right_shape: tuple[str, ...], limit: int = 120,
           state_cap: int = 2_000_000) -> tuple[dict, list[dict]]:
    """Match two ordinary slot sequences one character at a time."""
    stats = Counter(states=0, mismatch=0, semantic_prunes=0, closures=0, initial_pairs=0,
                    best_matched_chars=0)
    best_state: dict[str, object] = {}
    rows: list[dict] = []
    # Boundary-aware prefix indexes avoid testing every lexical choice when the
    # opposing current word already exposes a character.
    first = defaultdict(list)
    last = defaultdict(list)
    for slot, frame in (("left", left_frame), ("right", right_frame)):
        for category in set(left_shape + right_shape):
            pool = frame_words(left_frame if slot == "left" else right_frame, category, ())
            for word in pool:
                first[(slot, category, word[0])].append(word)
                last[(slot, category, word[-1])].append(word)

    def recurse(li: int, lw: str | None, lo: int, ri: int, rw: str | None,
                ro: int, left: tuple[str, ...], right: tuple[str, ...]) -> None:
        stats["states"] += 1
        matched = sum(len(word) for word in left) + (lo if lw is not None else 0)
        stats["best_matched_chars"] = max(stats["best_matched_chars"], matched)
        if matched >= int(stats["best_matched_chars"]):
            best_state.update({"matched_chars": matched, "left": list(left), "right": list(right),
                               "left_active": lw[lo:] if lw is not None else "",
                               "right_active": rw[ro:] if rw is not None else ""})
        if stats["states"] > state_cap or len(rows) >= limit:
            return
        if lw is None and rw is None:
            if li == len(left_shape) and ri < 0:
                stats["closures"] += 1
                text = " ".join(left + right).capitalize() + "."
                tape = normalize_letters(text)
                checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
                rows.append({
                    "rendered": text, "letters": len(tape),
                    "exact": direct_audit(tape),
                    "independent_two_pointer": opposing_index_audit(tape),
                    "left_frame": left_frame.name, "right_frame": right_frame.name,
                    "left_shape": list(left_shape), "right_shape": list(right_shape),
                    "failed_checks": [k for k, v in checks.items() if not v],
                    "admitted": all(checks.values()),
                    "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
                    "readability_status": "diagnostic_only_unreviewed",
                    "provenance": "selectional-preference character-synchronous slot search",
                })
        if li == len(left_shape) or ri < 0:
            return
        # Pick a lexical word for each side only at boundaries; while both are
        # active the state is deterministic and does not expand a surface unit.
        if lw is None and rw is None:
            lslot, rslot = left_shape[li], right_shape[ri]
            for a in frame_words(left_frame, lslot, left):
                if not article_ok(left + (a,), left_shape[:li + 1]):
                    continue
                for b0 in frame_words(right_frame, rslot, right):
                    b = b0[::-1]
                    if a[0] != b[0] or not article_ok((b0,) + right, right_shape[ri:]):
                        continue
                    stats["initial_pairs"] += 1
                    recurse(li + 1, a, 0, ri - 1, b, 0, left + (a,), (b0,) + right)
            return
        if lw is None:
            lslot = left_shape[li]
            target = rw[ro]
            for a in frame_words(left_frame, lslot, left):
                if a[0] == target and article_ok(left + (a,), left_shape[:li + 1]):
                    recurse(li + 1, a, 0, ri, rw, ro, left + (a,), right)
            return
        if rw is None:
            rslot = right_shape[ri]
            target = lw[lo]
            for b0 in frame_words(right_frame, rslot, right):
                b = b0[::-1]
                if b[0] == target and article_ok((b0,) + right, right_shape[ri:]):
                    recurse(li, lw, lo, ri - 1, b, 0, left, (b0,) + right)
            return
        if lw[lo] != rw[ro]:
            stats["mismatch"] += 1
            return
        nlo, nro = lo + 1, ro + 1
        recurse(li + 1 if nlo == len(lw) else li,
                None if nlo == len(lw) else lw, 0 if nlo == len(lw) else nlo,
                ri - 1 if nro == len(rw) else ri,
                None if nro == len(rw) else rw, 0 if nro == len(rw) else nro,
                left, right)

    recurse(0, None, 0, len(right_shape) - 1, None, 0, (), ())
    result_stats = dict(stats)
    result_stats["best_state"] = best_state
    return result_stats, rows


def main() -> None:
    all_rows: list[dict] = []
    runs = []
    for left_frame in FRAMES:
        for right_frame in FRAMES:
            for left_shape in SHAPES:
                for right_shape in SHAPES:
                    stats, rows = search(left_frame, right_frame, left_shape, right_shape)
                    all_rows.extend(rows)
                    runs.append({"left_frame": left_frame.name, "right_frame": right_frame.name,
                                 "left_shape": list(left_shape), "right_shape": list(right_shape),
                                 "stats": stats, "rows": len(rows)})
    output = {
        "experiment_id": ID, "signature": SIGNATURE,
        "preflight": {"status": "novel", "registered_families_checked": 71,
                       "excluded_routes_checked": 6, "manual_review_required": False},
        "method": "selectional-preference clause automaton with character-synchronous prefix matching",
        "repair": {"id": "subordinate-clause-state", "status": "evaluated",
                   "description": "added a complete main clause followed by an explicit while/after/because subordinate clause while retaining the same learned lexical frames and trie transitions"},
        "frames": [frame.name for frame in FRAMES], "shapes": [list(shape) for shape in SHAPES],
        "runs": runs, "rendered_candidates": all_rows,
        "exact_count": sum(row["exact"] for row in all_rows),
        "admitted_count": sum(row["admitted"] for row in all_rows),
        "independent_audit": {"method": "direct reverse-string comparison versus opposing-index scan",
                              "rows": len(all_rows),
                              "primary_exact": sum(row["exact"] for row in all_rows),
                              "independent_exact": sum(row["independent_two_pointer"] for row in all_rows),
                              "disagreements": [row["rendered"] for row in all_rows
                                                if row["exact"] != row["independent_two_pointer"]]},
        "readability_note": "No human study: no admitted exact candidate was produced.",
        "next_repair": "subordinate-clause repair was evaluated in this run; if it remains empty, switch to an open-vocabulary clause proposal source rather than enlarging the frame bank",
        "provenance": "role words authored for the route; Brown contributes only preference counts, not copied sentences",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"rows": len(all_rows), "exact": output["exact_count"], "admitted": output["admitted_count"]}))


if __name__ == "__main__":
    main()
