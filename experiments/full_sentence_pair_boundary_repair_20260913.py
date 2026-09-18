"""Invalidated POS-template diagnostic retained for experiment history.

This draft only combines POS-typed lexical menus with a shallow role replay.
That is insufficient to establish a connected grammatical discourse or
semantic coherence. Brown contributes only attested token/POS options; it
never contributes a copied sentence. This module must never admit a candidate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 180


@dataclass(frozen=True)
class Token:
    word: str
    role: str
    tag: str
    count: int
    source: str = "nltk.corpus.brown"


TAGS = {
    "det": ("AT", "DT", "DTI", "DTS", "DTX"),
    "adj": ("JJ", "JJR", "JJS"),
    "noun": ("NN", "NNS", "NP", "NPS"),
    "agent": ("NN", "NNS", "NP", "NPS"),
    "object": ("NN", "NNS", "NP", "NPS"),
    "verb": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "eventverb": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "prep": ("IN",),
}

SEMANTIC_WORDS = {
    "agent": {"artists", "children", "doctors", "editors", "farmers", "friends", "guards", "nurses", "parents", "teachers", "teams", "workers"},
    "object": {"archive", "book", "bridge", "door", "garden", "gate", "letter", "map", "message", "record", "report", "room", "stone", "story", "table", "task", "tool", "wall", "water"},
    "eventverb": {"builds", "carries", "closes", "finds", "keeps", "makes", "marks", "opens", "prepares", "records", "repairs", "uses", "writes"},
}


def compile_brown(limit: int = 36) -> dict[str, tuple[Token, ...]]:
    from nltk.corpus import brown
    counts: dict[str, Counter[tuple[str, str]]] = {r: Counter() for r in TAGS}
    for raw, tag in brown.tagged_words():
        word = raw.casefold()
        if not word.isascii() or not word.isalpha():
            continue
        for role, tags in TAGS.items():
            if tag in tags:
                counts[role][(word, tag)] += 1
    return {role: tuple(Token(w, role, tag, n) for (w, tag), n in c.most_common(limit)
                         if role not in SEMANTIC_WORDS or w in SEMANTIC_WORDS[role])
            for role, c in counts.items()}


# POS-shaped diagnostic channel only, not a connected discourse grammar.
LEFT = ("adj", "agent", "eventverb", "a", "adj", "object", "prep", "the", "object",
        "while", "the", "adj", "agent", "eventverb", "a", "adj", "object", "prep", "the", "object",
        "because", "the", "adj", "agent", "eventverb", "a", "object")
RIGHT = ("adj", "agent", "eventverb", "a", "adj", "object", "prep", "the", "object",
         "while", "the", "adj", "agent", "eventverb", "a", "adj", "object", "prep", "the", "object",
         "because", "the", "adj", "agent", "eventverb", "a", "object")
FIXED = {"the": ("the",), "a": ("a",), "while": ("while",), "because": ("because",)}


def debt(left: str, right: str) -> dict[str, object]:
    """Compare all currently exposed characters, independent of spaces."""
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    n = min(len(a), len(b))
    mismatch = next((i for i in range(n) if a[i] != b[i]), None)
    return {"compatible": mismatch is None, "matched": n if mismatch is None else mismatch,
            "mismatch": mismatch}


def render(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    # right is stored in normal grammatical order; it was merely emitted from
    # its final boundary inward during search.
    text = " ".join((*left, *right))
    return text[:1].upper() + text[1:] + "."


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    first = next(((i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    return {"letters": len(tape), "exact": bool(tape) and first is None, "first_mismatch": first}


def parse_witness(text: str, vocab: dict[str, tuple[Token, ...]]) -> dict[str, object]:
    """Independently replay the complete feature grammar and agreement facts."""
    words = tuple(text.casefold().rstrip(".").split())
    if len(words) != len(LEFT) + len(RIGHT):
        return {"complete": False, "reason": "token_count"}
    allowed = {role: {t.word for t in vals} for role, vals in vocab.items()}
    for i, role in enumerate(LEFT):
        if role in FIXED and words[i] != FIXED[role][0]: return {"complete": False, "reason": "left_fixed", "position": i}
        if role in TAGS and words[i] not in allowed[role]: return {"complete": False, "reason": "left_role", "position": i}
    offset = len(LEFT)
    for i, role in enumerate(RIGHT):
        word = words[offset + i]
        if role in FIXED and word != FIXED[role][0]: return {"complete": False, "reason": "right_fixed", "position": i}
        if role in TAGS and word not in allowed[role]: return {"complete": False, "reason": "right_role", "position": i}
    # Agent sets are plural and event verbs are third-person present, enforcing
    # the only agreement pattern admitted by this grammar.
    for base in (0, len(LEFT)):
        if not words[base + 1].endswith("s") or not words[base + 2].endswith("s"):
            return {"complete": False, "reason": "agreement", "position": base}
    return {"complete": True, "roles": {"agent": "plural", "eventverb": "third_person_present", "object": "lexically attested noun"}}


def search(*, beam: int = 96, limit: int = 80) -> dict[str, object]:
    vocab = compile_brown(limit)
    roles = [r for r in LEFT if r in TAGS]
    # State stores channels in their grammatical order; right is emitted from
    # its final grammatical role inward, but rendered in normal discourse order.
    frontier = [((), (), "", "", [])]
    rejections: list[dict[str, object]] = []
    emissions = 0
    for depth, (lrole, rrole) in enumerate(zip(LEFT, reversed(RIGHT))):
        choices_l = FIXED[lrole] if lrole in FIXED else tuple(t.word for t in vocab[lrole])
        choices_r = FIXED[rrole] if rrole in FIXED else tuple(t.word for t in vocab[rrole])
        nxt = []
        for left, right, lt, rt, prov in frontier:
            for lw in choices_l:
                for rw in choices_r:
                    emissions += 1
                    check = debt(" ".join((*left, lw)), " ".join((rw, *right)))
                    item = {"left_word": lw, "right_word": rw, "depth": depth, "check": check}
                    if check["compatible"]:
                        nxt.append(((*left, lw), (rw, *right), " ".join((*left, lw)), " ".join((rw, *right)), [*prov, item]))
                    elif len(rejections) < 240:
                        # POS-shaped complete diagnostics use the most-attested
                        # remaining lexical options; they are explicitly rejects.
                        def first(role: str) -> str:
                            return FIXED[role][0] if role in FIXED else vocab[role][0].word
                        tail_l = tuple(first(x) for x in LEFT[depth + 1:])
                        prefix_r = tuple(first(x) for x in RIGHT[:len(RIGHT) - depth - 1])
                        full_left, full_right = (*left, lw, *tail_l), (*prefix_r, rw, *right)
                        rendered = render(full_left, full_right)
                        rejections.append({"kind": "invalid_pos_template_rejected_at_emission",
                            "rendered": rendered, "depth": depth, "emission": item,
                            "independent_exact": independent_audit(rendered),
                            "grammar_parse": parse_witness(rendered, vocab),
                            "semantic_witness": {"events": "agent repairs object; observer records object; committee approves object", "links": ["while", "because"]},
                            "provenance": [*prov, item]})
        nxt.sort(key=lambda s: -sum(x["check"]["matched"] for x in s[4]))
        frontier = nxt[:beam]
        if not frontier:
            break
    closures = []
    for left, right, _, _, provenance in frontier:
        rendered = render(left, right)
        checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
        closures.append({"rendered": rendered, "letters": len(normalize_letters(rendered)),
                         "checks": checks, "independent_exact": independent_audit(rendered),
                         "grammar_parse": parse_witness(rendered, vocab),
                         "semantic_witness": {"events": "agent repairs object; observer records object; committee approves object", "links": ["while", "because"]},
                         "provenance": provenance})
    admitted: list[dict[str, object]] = []
    return {"status": "invalidated_missing_connected_feature_grammar",
            "method": "full_sentence_pair_boundary_repair", "corpus": "nltk.corpus.brown",
            "grammar": {"left_roles": LEFT, "right_roles": RIGHT, "semantic_witness": "agent repairs object while observer records object because committee approves object"},
            "parameters": {"beam": beam, "limit_per_role": limit, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS},
            "stats": {"emissions": emissions, "surviving_states": len(frontier), "complete_grammar_rejections": len(rejections), "admitted": len(admitted)},
            "closures": closures, "rejections": rejections, "admitted": admitted,
            "candidate_use": "forbidden: this is a POS-template diagnostic, not a connected grammar witness",
            "reader_facing_next_test": "None. These rows are ineligible for a reader package.",
            "next_repair": "Use an explicitly recursive feature grammar with independently parsed tree structure, agreement, valency, and semantic role bindings; lexical channels may only constrain its leaves."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--beam", type=int, default=96)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--output", type=Path, default=ROOT / "runs" / "full-sentence-pair-boundary-repair-2026-09-13" / "result.json")
    a = p.parse_args()
    result = search(beam=a.beam, limit=a.limit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
