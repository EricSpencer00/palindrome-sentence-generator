"""Seedless full-sequence grammar product with a seam inside a clause.

The product chooses the outer grammar slots first and compares their characters
immediately.  It never builds a finished sentence and then reverses or
resegments it.  Unlike the clause-pair lanes, the two pointers walk one
complete grammar sequence, so the palindrome seam may land in the middle of a
verb, adverb, or word boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "full-sequence-grammar-product-20260917.json"
EXPERIMENT_ID = "full-sequence-grammar-product-20260917"
SIGNATURE = (
    "seedless-full-sequence-slot-product|outer-slot-character-equation|"
    "center-inside-clause|typed-chain-grammar|independent-audit"
)

# These are hand-authored role banks, not a catalogue of completed sentences.
# The known 51-letter palindrome is admitted only as a quarantined engine
# fixture below, never as a generated result.
BANKS: dict[str, tuple[str, ...]] = {
    "NOUN": (
        "animal", "apple", "artist", "atlas", "baker", "boat", "book",
        "bread", "bridge", "cat", "chair", "child", "cod", "coin",
        "comet", "crab", "crow", "day", "dog", "doc", "door", "dream",
        "drum", "dust", "earth", "fact", "farm", "fast", "fatness",
        "field", "film", "fish", "flag", "flower", "food", "gate", "gift",
        "girl", "glass", "grain", "hand", "heart", "hill", "home", "horse",
        "idea", "island", "jar", "kid", "king", "lake", "lamp", "letter",
        "life", "light", "line", "map", "meal", "memo", "message", "mind",
        "moon", "name", "note", "ocean", "page", "pan", "park", "path",
        "plan", "poem", "pond", "pot", "rain", "record", "river", "road",
        "room", "rose", "rule", "salt", "school", "sea", "seed", "ship",
        "shore", "sky", "song", "star", "stone", "story", "sun", "table",
        "task", "thing", "time", "town", "tree", "truth", "wall", "water",
        "wave", "wind", "word", "world",
    ),
    "VERB_BASE": (
        "admire", "answer", "ask", "bake", "call", "carry", "catch", "check",
        "clean", "close", "cook", "copy", "cross", "cut", "dance", "deliver",
        "deny", "diet", "dissent", "draw", "dream", "drink", "drive", "eat",
        "edit", "enter", "fast", "find", "fix", "fold", "follow", "give",
        "guard", "guide", "help", "hold", "hope", "keep", "know", "label",
        "learn", "leave", "like", "live", "love", "make", "mark", "meet",
        "mend", "name", "note", "open", "paint", "pay", "plan", "plant",
        "play", "read", "record", "repair", "repay", "rest", "ride", "ring",
        "save", "see", "send", "serve", "share", "sing", "speak", "spell",
        "start", "stay", "stop", "store", "study", "take", "teach", "tend",
        "test", "thank", "think", "tie", "travel", "treat", "use", "visit",
        "wait", "walk", "want", "watch", "write",
    ),
    "VERB_S": (
        "admires", "answers", "asks", "bakes", "calls", "carries", "catches",
        "checks", "cleans", "closes", "cooks", "copies", "crosses", "cuts",
        "delivers", "denies", "eats", "edits", "enters", "fasts", "finds",
        "follows", "gives", "guards", "guides", "helps", "holds", "hopes",
        "keeps", "knows", "labels", "learns", "leaves", "likes", "lives",
        "loves", "makes", "marks", "meets", "mends", "names", "notes", "opens",
        "paints", "pays", "plans", "plants", "plays", "prevents", "reads",
        "records", "repairs", "repays", "rests", "rides", "rings", "saves",
        "sees", "sends", "serves", "shares", "sings", "speaks", "starts",
        "stays", "stops", "stores", "studies", "takes", "teaches", "tends",
        "tests", "thanks", "thinks", "ties", "travels", "treats", "uses",
        "visits", "waits", "walks", "wants", "watches", "writes",
    ),
    "PRON": ("i", "we", "you", "he", "she", "it", "they"),
    "DET": ("a", "an", "the", "some", "no", "one", "each", "every", "this", "that"),
    "ADV": ("again", "always", "often", "never", "now", "not", "once", "only", "perhaps", "quite", "rather", "still", "then", "today", "too", "very", "well"),
    "PREP": ("about", "after", "at", "by", "for", "from", "in", "into", "near", "of", "on", "over", "through", "to", "under", "with"),
    "ADJ": ("fast", "fat", "good", "kind", "new", "old", "quiet", "small", "true", "warm", "young"),
    "CONJ": ("and", "or", "but"),
}

PATTERNS: dict[str, tuple[str, ...]] = {
    # This is a productive chain grammar.  Its familiar catalogue instance is
    # used only as an implementation oracle, and is excluded from admission.
    "chain": ("NOUN", "VERB_BASE", "PRON", "VERB_BASE", "DET", "ADJ", "ADV", "VERB_S", "DET", "NOUN", "PRON", "VERB_BASE", "PREP", "NOUN"),
    "chain_no_modifier": ("NOUN", "VERB_BASE", "PRON", "VERB_BASE", "DET", "NOUN", "ADV", "VERB_S", "DET", "NOUN", "PRON", "VERB_BASE", "PREP", "NOUN"),
    "paired_clauses": ("DET", "ADJ", "NOUN", "VERB_S", "DET", "NOUN", "PREP", "DET", "NOUN", "PRON", "VERB_BASE", "DET", "NOUN"),
    "coordinated": ("DET", "NOUN", "VERB_S", "DET", "NOUN", "CONJ", "PRON", "VERB_BASE", "DET", "NOUN", "PREP", "DET", "NOUN"),
}

FUNCTION_WORDS = {
    "a", "an", "the", "some", "no", "one", "each", "every", "this", "that",
    "i", "we", "you", "he", "she", "it", "they", "about", "after", "at", "by",
    "for", "from", "in", "into", "near", "of", "on", "over", "through", "to",
    "under", "with", "and", "or", "but",
}

CATALOGUE_FIXTURE = (
    "doc", "note", "i", "dissent", "a", "fast", "never", "prevents",
    "a", "fatness", "i", "diet", "on", "cod",
)
CATALOGUE_TAPE = "".join(CATALOGUE_FIXTURE)


def letters(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalpha())


def exact_audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def anti_shortcut(words: tuple[str, ...]) -> dict:
    content = [word for word in words if word not in FUNCTION_WORDS]
    tape = "".join(words)
    return {
        "repeated_content": len(content) != len(set(content)),
        "self_palindromic_words": [word for word in words if len(word) > 1 and word == word[::-1]],
        "word_order_symmetry": list(words) == [word[::-1] for word in reversed(words)],
        "catalogue_tape": tape == CATALOGUE_TAPE,
        "catalogue_word_overlap": sorted(set(content) & (set(CATALOGUE_FIXTURE) - FUNCTION_WORDS)),
    }


def _eligible(word: str, used: frozenset[str]) -> bool:
    return word not in used or word in FUNCTION_WORDS


@dataclass
class SearchResult:
    paths: list[dict]
    states: int
    mismatch_edges: int
    budget_exhausted: bool
    longest_partial: dict | None
    mismatch_frontiers: list[dict]


def search_pattern(slots: tuple[str, ...], *, state_budget: int = 250_000,
                   catalogue_fixture: bool = False,
                   forbidden_words: frozenset[str] = frozenset()) -> SearchResult:
    """Search one full slot sequence from both outer ends.

    `assign` is carried in every state, so backpointers cannot silently turn
    into a finished-tape Cartesian product.  A character mismatch is rejected
    before either side can advance to a new slot.
    """
    n = len(slots)
    stack = [(0, n - 1, None, 0, None, 0, (None,) * n, frozenset())]
    seen: set[tuple] = set()
    paths: list[dict] = []
    mismatch_edges = 0
    mismatch_frontiers: list[dict] = []
    states = 0
    longest_partial: dict | None = None
    while stack and states < state_budget:
        li, ri, left_word, left_pos, right_word, right_pos, assign, used = stack.pop()
        states += 1
        # Word boundaries are epsilon transitions on the two independent sides.
        if left_word is not None and left_pos == len(left_word):
            stack.append((li + 1, ri, None, 0, right_word, right_pos, assign, used))
            continue
        if right_word is not None and right_pos == len(right_word):
            stack.append((li, ri - 1, left_word, left_pos, None, 0, assign, used))
            continue
        key = (li, ri, left_word, left_pos, right_word, right_pos, assign, used)
        if key in seen:
            continue
        seen.add(key)
        matched = sum(1 for word in assign if word is not None)
        if longest_partial is None or matched > longest_partial["slots_realized"]:
            longest_partial = {"slots_realized": matched, "slots": slots, "assignment": assign}
        # The two pointers can meet inside one lexical slot.  In that case the
        # active word belongs to only one side: its remaining prefix/suffix is
        # the center of the whole tape.  Requiring the word itself to be a
        # palindrome would incorrectly reject the known `... never prevents
        # ...` geometry, whose center is the single `p` in `prevents`.
        if li == ri and left_word is None and right_word is not None:
            if len(right_word) - right_pos <= 1:
                words = tuple(word for word in assign if word is not None)
                rendered = " ".join(words)
                audit = exact_audit(rendered)
                shortcut = anti_shortcut(words)
                if audit["exact"] and (catalogue_fixture or not any(shortcut.values())):
                    paths.append({"rendered": rendered, "words": words, "audit": audit,
                                  "anti_shortcut": shortcut, "provenance": {"slots": slots,
                                  "outer_slot_expansion": True, "live_character_edges": True,
                                  "center_inside_slot": True,
                                  "catalogue_fixture": catalogue_fixture}})
            continue
        if li == ri and right_word is None and left_word is not None:
            if len(left_word) - left_pos <= 1:
                words = tuple(word for word in assign if word is not None)
                rendered = " ".join(words)
                audit = exact_audit(rendered)
                shortcut = anti_shortcut(words)
                if audit["exact"] and (catalogue_fixture or not any(shortcut.values())):
                    paths.append({"rendered": rendered, "words": words, "audit": audit,
                                  "anti_shortcut": shortcut, "provenance": {"slots": slots,
                                  "outer_slot_expansion": True, "live_character_edges": True,
                                  "center_inside_slot": True,
                                  "catalogue_fixture": catalogue_fixture}})
            continue
        if li > ri:
            words = tuple(word for word in assign if word is not None)
            rendered = " ".join(words)
            audit = exact_audit(rendered)
            shortcut = anti_shortcut(words)
            if audit["exact"] and (catalogue_fixture or not any(shortcut.values())):
                paths.append({"rendered": rendered, "words": words, "audit": audit,
                              "anti_shortcut": shortcut, "provenance": {"slots": slots,
                              "outer_slot_expansion": True, "live_character_edges": True,
                              "catalogue_fixture": catalogue_fixture}})
            continue
        if li == ri and left_word is None and right_word is None:
            for word in BANKS[slots[li]]:
                if word in forbidden_words or not _eligible(word, used):
                    continue
                if len(letters(word)) > 1:
                    continue
                updated = list(assign); updated[li] = word
                stack.append((li + 1, ri - 1, None, 0, None, 0, tuple(updated),
                              used | ({word} if word not in FUNCTION_WORDS else set())))
            continue
        if left_word is None:
            for word in BANKS[slots[li]]:
                if word in forbidden_words or not _eligible(word, used):
                    continue
                updated = list(assign); updated[li] = word
                stack.append((li, ri, word, 0, right_word, right_pos, tuple(updated),
                              used | ({word} if word not in FUNCTION_WORDS else set())))
            continue
        if right_word is None:
            for word in BANKS[slots[ri]]:
                if word in forbidden_words or not _eligible(word, used):
                    continue
                updated = list(assign); updated[ri] = word
                stack.append((li, ri, left_word, left_pos, word, 0, tuple(updated),
                              used | ({word} if word not in FUNCTION_WORDS else set())))
            continue
        left_tape, right_tape = letters(left_word), letters(right_word)
        if left_tape[left_pos] != right_tape[-1 - right_pos]:
            mismatch_edges += 1
            if len(mismatch_frontiers) < 24:
                mismatch_frontiers.append({
                    "left_slot": li, "left_role": slots[li], "left_word": left_word,
                    "left_position": left_pos, "left_char": left_tape[left_pos],
                    "right_slot": ri, "right_role": slots[ri], "right_word": right_word,
                    "right_position_from_end": right_pos, "right_char": right_tape[-1 - right_pos],
                    "assignment": assign,
                })
            continue
        stack.append((li, ri, left_word, left_pos + 1, right_word, right_pos + 1, assign, used))
    return SearchResult(paths, states, mismatch_edges, bool(stack), longest_partial, mismatch_frontiers)


def seam_repair_menu(frontier: dict, *, forbidden_words: frozenset[str] = frozenset()) -> dict:
    """Propose one-slot alternatives that preserve the matched edge prefix.

    This is deliberately a menu, not an admission gate: the resumed product
    must still close the entire tape before anything can be rendered.  Keeping
    the already-matched prefix/suffix fixed prevents the repair from becoming a
    fresh Cartesian sweep from the root.
    """
    left_prefix = letters(frontier["left_word"])[:frontier["left_position"]]
    right_suffix = letters(frontier["right_word"])[::-1][:frontier["right_position_from_end"]]
    left_char = frontier["right_char"]
    right_char = frontier["left_char"]
    left_alternatives = []
    for word in BANKS[frontier["left_role"]]:
        tape = letters(word)
        if word in forbidden_words or not tape.startswith(left_prefix):
            continue
        if len(tape) > frontier["left_position"] and tape[frontier["left_position"]] == left_char:
            left_alternatives.append(word)
    right_alternatives = []
    for word in BANKS[frontier["right_role"]]:
        tape = letters(word)[::-1]
        if word in forbidden_words or not tape.startswith(right_suffix):
            continue
        if len(tape) > frontier["right_position_from_end"] and tape[frontier["right_position_from_end"]] == right_char:
            right_alternatives.append(word)
    return {
        "left_slot": frontier["left_slot"], "right_slot": frontier["right_slot"],
        "preserved_left_prefix": left_prefix, "preserved_right_suffix": right_suffix,
        "left_alternatives": left_alternatives[:50], "right_alternatives": right_alternatives[:50],
        "resume_required": True,
    }


def render_fixture() -> dict:
    text = "Doc, note: I dissent. A fast never prevents a fatness. I diet on cod."
    audit = exact_audit(text)
    return {"rendered": text, "words": CATALOGUE_FIXTURE, "audit": audit,
            "status": "quarantined_catalogue_fixture", "admitted": False,
            "provenance": "published catalogue oracle; implementation test only"}


def run() -> dict:
    fixture = render_fixture()
    searches: dict[str, dict] = {}
    novel: list[dict] = []
    for name, slots in PATTERNS.items():
        result = search_pattern(slots, forbidden_words=frozenset(CATALOGUE_FIXTURE))
        searches[name] = {"states": result.states, "mismatch_edges": result.mismatch_edges,
                          "budget_exhausted": result.budget_exhausted,
                          "exact_paths": len(result.paths),
                          "longest_partial": result.longest_partial,
                          "mismatch_frontiers": result.mismatch_frontiers,
                          "seam_repair_menus": [seam_repair_menu(frontier,
                              forbidden_words=frozenset(CATALOGUE_FIXTURE))
                              for frontier in result.mismatch_frontiers[:3]],
                          "paths": result.paths[:20]}
        novel.extend(result.paths)
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_novel_exact_closure" if not novel else "completed_exact_paths_need_readers",
        "method": "seedless full-sequence typed slot product; outer grammar slots and character edges are expanded synchronously",
        "fixture": fixture,
        "searches": searches,
        "novel_exact_candidates": novel[:50],
        "reader_gate": "closed; no novel exact survivor means no human study yet",
        "next_repair": "typed seam repair at the first mismatching edge, changing one role bank while preserving all earlier matched edges",
        "independent_audits": ["two-pointer letter comparison", "forward/reverse SHA-256"],
        "anti_shortcut_policy": ["no finished-tape reversal", "no nested palindrome core", "no repeated content word", "catalogue fixture never admitted"],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    result = json.loads(OUT.read_text())
    print(json.dumps({"status": result["status"], "novel_exact": len(result["novel_exact_candidates"]),
                      "searches": {k: (v["states"], v["exact_paths"]) for k, v in result["searches"].items()}}))
