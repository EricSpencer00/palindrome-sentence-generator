"""Live-buffer whole-sentence construction.

The earlier slot lane indexed the newest right word even when the right-side
character debt still belonged to an older word.  That is an unsound shortcut:
it pruned the known 38-letter control before reaching its centre.  This lane
keeps the complete unmatched prefix/suffix buffers and compares only the
characters whose opposite endpoints are already assigned.  Words are chosen
on both sides of the same grammar derivation; no finished tape is reversed or
repaired.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "live-buffer-grammar-search-20260919.json"
EXPERIMENT_ID = "live-buffer-grammar-search-20260919"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(words: tuple[str, ...] | list[str] | str) -> dict[str, object]:
    text = words if isinstance(words, str) else " ".join(words)
    tape = letters(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    mismatches = [
        (i, len(tape) - 1 - i)
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


@dataclass(frozen=True)
class Slot:
    role: str
    words: tuple[str, ...]


def compatible(prefix: str, suffix: str) -> bool:
    """Check assigned endpoints without assuming word-boundary alignment."""
    overlap = min(len(prefix), len(suffix))
    return prefix[:overlap] == suffix[::-1][:overlap]


def _distinct(word: str, left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return word not in left and word not in right and word != word[::-1]


def search(template: tuple[Slot, ...], *, limit: int = 64,
           state_limit: int = 500_000) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []
    states = 0
    pruned = 0
    boundary_index_hits = 0

    def witness(words: tuple[str, ...], prefix: str, suffix: str, depth: int) -> None:
        if len(witnesses) >= 24:
            return
        rendered = " ".join(words)
        witnesses.append({
            "rendered": rendered,
            "audit": independent_audit(rendered),
            "depth": depth,
            "prefix_letters": len(prefix),
            "suffix_letters": len(suffix),
            "reader_status": "diagnostic witness; not a complete candidate",
        })

    def add_candidate(words: tuple[str, ...], depth: int) -> None:
        checked = independent_audit(words)
        if not checked["exact"] or len(set(words)) != len(words):
            return
        candidates.append({
            "rendered": " ".join(words),
            "audit": checked,
            "provenance": {
                "construction": "joint whole-sentence live-buffer grammar",
                "template_roles": [slot.role for slot in template],
                "depth": depth,
                "finished_tape_reversal": False,
                "post_hoc_repair": False,
                "catalogue_text": False,
                "aligned_token_mirror": False,
                "repeated_word": False,
            },
            "reader_status": "unreviewed; exactness does not certify readability",
        })

    def walk(lo: int, hi: int, prefix: str, suffix: str,
             left: tuple[str, ...], right: tuple[str, ...], shifted: bool) -> None:
        nonlocal states, pruned, boundary_index_hits
        if states >= state_limit or len(candidates) >= limit:
            return
        if lo > hi:
            states += 1
            add_candidate(left + right, len(left))
            return
        if lo == hi:
            for word in template[lo].words:
                if not _distinct(word, left, right):
                    continue
                new_prefix = prefix + letters(word)
                if not compatible(new_prefix, suffix):
                    pruned += 1
                    witness(left + (word,) + right, new_prefix, suffix, len(left) + 1)
                    continue
                states += 1
                add_candidate(left + (word,) + right, len(left) + 1)
            return

        rights = template[hi].words
        # This index is safe only when the buffers are equal: then the next
        # left character is paired with the final character of the new right
        # word.  When buffers differ, no word-local index is sound.
        indexed: dict[str, tuple[str, ...]] | None = None
        if len(prefix) == len(suffix):
            by_last: dict[str, list[str]] = {}
            for word in rights:
                by_last.setdefault(letters(word)[-1], []).append(word)
            indexed = {key: tuple(value) for key, value in by_last.items()}

        for left_word in template[lo].words:
            if not _distinct(left_word, left, right):
                continue
            choices = rights
            if indexed is not None:
                choices = indexed.get(letters(left_word)[0], ())
                boundary_index_hits += 1
            for right_word in choices:
                if not _distinct(right_word, left, right) or right_word == left_word:
                    continue
                new_prefix = prefix + letters(left_word)
                new_suffix = letters(right_word) + suffix
                states += 1
                if not compatible(new_prefix, new_suffix):
                    pruned += 1
                    witness(left + (left_word,) + (right_word,) + right,
                            new_prefix, new_suffix, len(left) + 1)
                    continue
                walk(lo + 1, hi - 1, new_prefix, new_suffix,
                     left + (left_word,), (right_word,) + right,
                     shifted or letters(left_word) != letters(right_word)[::-1])

    walk(0, len(template) - 1, "", "", (), (), False)
    candidates.sort(key=lambda row: row["audit"]["letters"], reverse=True)
    return {
        "candidates": candidates[:limit],
        "witnesses": witnesses,
        "stats": {
            "states": states,
            "pruned": pruned,
            "boundary_index_hits": boundary_index_hits,
            "exact": len(candidates),
        },
    }


def _slot(role: str, *words: str) -> Slot:
    return Slot(role, tuple(dict.fromkeys(word.casefold() for word in words)))


def build_templates() -> dict[str, tuple[Slot, ...]]:
    # The first template is a complete, ordinary two-clause control.  The
    # control is included as a regression, not as a new result.
    seed = tuple(
        _slot(role, word)
        for role, word in (
            ("determiner", "an"), ("agent", "aide"), ("verb", "rips"),
            ("quantity", "nine"), ("object", "memos"), ("determiner", "some"),
            ("subject", "men"), ("verb", "inspire"), ("name", "diana"),
        )
    )
    determiners = _slot("determiner", "a", "an", "the", "some", "one", "this")
    agents = _slot(
        "agent", "aide", "poet", "bard", "clerk", "nurse", "pilot", "sailor",
        "scribe", "artist", "farmer", "teacher", "writer", "reader", "keeper",
        "author", "editor", "doctor", "dancer", "singer", "rider", "mother",
        "father", "sister", "brother", "son", "daughter",
    )
    verbs = _slot(
        "verb", "aids", "asks", "bakes", "calls", "carries", "draws", "edits",
        "feeds", "finds", "gives", "guides", "helps", "keeps", "marks", "meets",
        "mends", "names", "notes", "opens", "paints", "plays", "reads", "records",
        "rips", "saves", "sees", "sends", "shares", "sings", "speaks", "spells",
        "starts", "takes", "teaches", "tells", "tests", "thanks", "ties", "trims",
        "uses", "visits", "waits", "washes", "weaves", "writes",
    )
    quantities = _slot("quantity", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "many")
    objects = _slot(
        "object", "memo", "memos", "letter", "letters", "note", "notes", "book",
        "books", "poem", "poems", "map", "maps", "text", "texts", "tale", "tales",
        "chart", "charts", "story", "stories", "verse", "verses", "song", "songs",
        "sign", "signs", "plan", "plans", "list", "lists", "idea", "ideas", "message",
        "messages", "record", "records", "secret", "secrets", "answer", "answers",
    )
    subjects = _slot(
        "subject", "men", "women", "poets", "bards", "clerks", "nurses", "pilots",
        "sailors", "scribes", "artists", "farmers", "teachers", "writers", "readers",
        "keepers", "authors", "editors", "doctors", "dancers", "singers", "riders",
        "drivers", "friends", "mothers", "fathers", "sisters", "brothers", "sons",
        "daughters", "people",
    )
    plural_verbs = _slot(
        "verb", "admire", "aid", "ask", "bake", "call", "carry", "draw", "edit",
        "feed", "find", "give", "guide", "help", "keep", "mark", "meet", "mend",
        "name", "note", "open", "paint", "play", "read", "record", "rip", "save",
        "see", "send", "share", "sing", "speak", "spell", "start", "take", "teach",
        "tell", "test", "thank", "tie", "trim", "use", "visit", "wait", "wash",
        "weave", "write", "inspire", "honor",
    )
    names = _slot(
        "name", "diana", "nora", "maria", "anna", "ada", "eve", "ava", "noel", "leon",
        "iris", "elba", "otto", "oscar", "amelia", "aria", "clara", "david", "daniel",
        "edgar", "ellen", "eric", "helen", "isabel", "julia", "liam", "lena", "maya",
        "milo", "nina", "olivia", "paul", "sara", "simon", "susan", "talia", "victor",
        "will", "zoe",
    )
    expanded = (
        determiners, agents, verbs, quantities, objects,
        determiners, subjects, plural_verbs, names,
    )
    return {"seed_control": seed, "expanded_two_clause": expanded}


def run() -> dict[str, object]:
    templates = build_templates()
    searches = {name: search(template) for name, template in templates.items()}
    candidates = [
        {**candidate, "template": name}
        for name, result in searches.items()
        for candidate in result["candidates"]
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "joint typed grammar with complete live prefix/suffix buffers",
        "searches": searches,
        "candidates": candidates,
        "exact_candidates": [row for row in candidates if row["audit"]["exact"]],
        "best_length": max((row["audit"]["letters"] for row in candidates), default=0),
        "provenance": {
            "lexicon": "hand-authored ordinary English role banks",
            "generated_compositionally": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "catalogue_text": False,
            "reader_evidence": False,
        },
        "reader_gate": "closed until a complete exact candidate is independently blinded-rated",
        "next_construction": "add a new complete clause grammar with live buffers; do not repair a failed tape",
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    result = json.loads(OUT.read_text())
    print(json.dumps({
        "best_length": result["best_length"],
        "exact": len(result["exact_candidates"]),
        "searches": {key: value["stats"] for key, value in result["searches"].items()},
    }))
