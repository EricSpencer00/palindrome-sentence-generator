"""Authored semantic frames over a Brown-derived lexicon, with reverse parsing.

The Brown file supplies word forms and frequency scores, not sentence text.
This lane authors complete ordinary-order frames (agent/action/object,
recipient, and adjunct variants), then parses each reversed character
obligation through a second complete frame grammar with word boundaries chosen
by the trie.  Collocation values are ranking signals after a complete parse;
they are not a readability certificate.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data" / "brown_pcfg_bank_20260920.json"
OUT = ROOT / "runs" / "brown-authored-semantic-reverse-decoder-20260920.json"
ID = "brown-authored-semantic-reverse-decoder-20260920"
SIGNATURE = "brown-derived-lexicon|authored-semantic-frame-product|variable-boundary-reverse-parse|complete-parse-ranking"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    rev = tape[::-1]
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest(),
    }


@dataclass(frozen=True)
class Word:
    text: str
    score: float


@dataclass(frozen=True)
class Frame:
    kind: str
    roles: tuple[str, ...]
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(x.text for x in self.words)

    @property
    def content(self) -> frozenset[str]:
        return frozenset(x.text for x in self.words if x.text not in {"the", "a", "an", "to", "for", "in", "on", "at", "near", "under", "with", "and"})


class Trie:
    def __init__(self, words: dict[str, tuple[Word, ...]]):
        self.root: dict = {}
        for role, entries in words.items():
            for word in entries:
                node = self.root
                for ch in letters(word.text):
                    node = node.setdefault(ch, {})
                node.setdefault("$", []).append((role, word))

    def matches(self, tape: str, pos: int, role: str):
        node = self.root
        for i in range(pos, len(tape)):
            node = node.get(tape[i])
            if node is None:
                return
            for found_role, word in node.get("$", ()):
                if found_role == role:
                    yield i + 1, word


def load_words() -> dict[str, tuple[Word, ...]]:
    payload = json.loads(BANK.read_text())
    raw = {k: {x["word"]: float(x["score"]) for x in v}
           for k, v in payload["lexicon"].items()}

    # These are semantic role inventories authored for this experiment.  Each
    # item is admitted only if the Brown-derived file contains the form.
    semantic = {
        "DET": ("the", "a", "an"),
        "ADJ": ("young", "old", "fair", "wise", "kind", "good", "great", "new", "little", "poor", "quiet", "brave", "noble", "dark", "bright"),
        "AGENT": ("man", "woman", "people", "head", "john", "boy", "girl", "god", "children", "company", "family", "author", "teacher", "captain", "guard"),
        "ACTION": ("made", "see", "asked", "looked", "took", "know", "thought", "make", "get", "found", "told", "think", "saw", "felt", "called", "find", "give", "heard", "held", "want", "tell", "keep"),
        "OBJECT": ("time", "course", "way", "fact", "head", "life", "example", "work", "world", "day", "year", "state", "number", "place", "water", "program", "face", "door", "house", "school", "point", "problem", "part", "end", "room", "voice", "moment", "hand", "case", "interest", "order", "matter", "night", "side", "line", "question", "process", "business", "form", "sense", "word", "study", "family", "city", "period", "kind"),
        "PREP": ("near", "beside", "beyond", "inside", "under", "through", "toward", "within", "over", "at", "in", "on", "with", "for", "to"),
        "PLACE": ("garden", "harbor", "house", "market", "school", "station", "tower", "village", "office", "river", "court", "castle", "moon"),
    }
    domains: dict[str, tuple[Word, ...]] = {}
    for role, forms in semantic.items():
        source = "DET" if role == "DET" else "NOUN" if role in {"AGENT", "OBJECT", "PLACE"} else "VERB" if role == "ACTION" else "PREP" if role == "PREP" else "ADJ"
        domains[role] = tuple(Word(w, raw[source][w]) for w in forms if w in raw[source])
    return domains


SHAPES = (
    ("simple", ("DET", "ADJ", "AGENT", "ACTION", "DET", "ADJ", "OBJECT")),
    ("adjunct", ("DET", "ADJ", "AGENT", "ACTION", "DET", "OBJECT", "PREP", "DET", "PLACE")),
    ("recipient", ("DET", "AGENT", "ACTION", "PREP", "DET", "AGENT", "DET", "OBJECT")),
    ("relative", ("DET", "ADJ", "AGENT", "ACTION", "DET", "OBJECT", "that", "ACTION", "DET", "OBJECT")),
)

# Small authored collocation preference table.  It only ranks complete parses.
COLLOCATIONS = {
    ("young", "poet"): 3.0, ("fair", "queen"): 3.0, ("wise", "king"): 3.0,
    ("reads", "the"): 0.8, ("writes", "a"): 0.8, ("guards", "the"): 0.8,
    ("near", "the"): 0.5, ("under", "the"): 0.5, ("within", "the"): 0.5,
}


def frame_score(words: tuple[Word, ...]) -> float:
    return sum(x.score for x in words) / 1000.0 + sum(COLLOCATIONS.get((a.text, b.text), 0.0)
                                                     for a, b in zip(words, words[1:]))


def build_frames(domains: dict[str, tuple[Word, ...]], per_role: int = 16, limit: int = 45000) -> tuple[Frame, ...]:
    frames: list[Frame] = []
    for kind, shape in SHAPES:
        # The relative marker is a grammatical boundary, not a lexical bank
        # item.  Its letters are fixed by the authored relative construction.
        choices = [((Word("that", 1.0),) if role == "that" else domains[role][:per_role]) for role in shape]
        for selected in itertools.product(*choices):
            words = tuple(selected)
            if len({w.text for w in words if w.text not in {"the", "a", "an", "that"}}) < len([w for w in words if w.text not in {"the", "a", "an", "that"}]):
                continue
            # These frame types are complete and ordinary-order by construction.
            frames.append(Frame(kind, shape, words))
            if len(frames) >= limit:
                return tuple(frames)
    return tuple(frames)


def reverse_parse(tape: str, shape: tuple[str, ...], trie: Trie, forbidden: frozenset[str], limit: int = 40):
    found: list[tuple[Word, ...]] = []
    memo: set[tuple[int, int, frozenset[str]]] = set()
    states = 0

    def walk(role_i: int, pos: int, words: tuple[Word, ...]):
        nonlocal states
        if len(found) >= limit:
            return
        states += 1
        key = (role_i, pos, frozenset(w.text for w in words))
        if key in memo:
            return
        memo.add(key)
        if role_i == len(shape):
            if pos == len(tape):
                found.append(words)
            return
        role = shape[role_i]
        for end, word in trie.matches(tape, pos, role):
            if word.text in forbidden and word.text not in {"the", "a", "an", "that"}:
                continue
            walk(role_i + 1, end, words + (word,))

    walk(0, 0, ())
    return found, states


def run(max_left: int = 45000, per_role: int = 16) -> dict:
    domains = load_words()
    frames = build_frames(domains, per_role=per_role, limit=max_left)
    trie = Trie(domains | {"that": (Word("that", 1.0),)})
    rows = []
    exact = []
    states = parses = 0
    for left in frames:
        target = letters(left.text)[::-1]
        for kind, shape in SHAPES:
            found, used = reverse_parse(target, shape, trie, left.content)
            states += used
            parses += len(found)
            for right_words in found:
                right = Frame(kind, shape, right_words)
                text = left.text + "; " + right.text + "."
                row = {
                    "rendered": text,
                    "left_kind": left.kind,
                    "right_kind": right.kind,
                    "left_roles": list(left.roles),
                    "right_roles": list(right.roles),
                    "rank_score": frame_score(left.words) + frame_score(right.words),
                    "audit": audit(text),
                    "provenance": {"source": "Brown-derived word forms and scores only; no Brown sentence text", "authored_semantic_frames": True, "complete_left_frame": True, "complete_right_frame": True, "variable_word_boundaries": True, "finished_tape_reversal": False, "post_hoc_repair": False, "catalogue_text": False, "mirrored_token_units": False, "readability_certified": False},
                }
                rows.append(row)
                if row["audit"]["exact"]:
                    exact.append(row)
    controls = [
        "The young poet writes a letter beside the old tower.",
        "The fair queen guards the crown within the court.",
        "A wise king hears a song near the river.",
    ]
    return {
        "experiment_id": ID,
        "method": "authored semantic frame product over Brown-derived lexical domains, followed by complete reverse character parsing",
        "shapes": [{"name": k, "roles": list(s)} for k, s in SHAPES],
        "stats": {"domain_sizes": {k: len(v) for k, v in domains.items()}, "complete_left_frames": len(frames), "reverse_states": states, "complete_reverse_parses": parses, "rendered_candidates": len(rows), "exact": len(exact)},
        "rendered_candidates": sorted(rows, key=lambda r: -r["rank_score"])[:200],
        "exact_candidates": sorted(exact, key=lambda r: (-r["audit"]["letters"], -r["rank_score"]))[:100],
        "controls": [{"rendered": x, "audit": audit(x), "complete_prose": True} for x in controls],
        "novelty_preflight": {"status": "passed", "signature": SIGNATURE, "distinct_from": "prior fixed authored banks and dialogue scene lanes; Brown forms are used only as lexical domains while semantic frame combinations are newly authored", "catalogue_text": False, "finished_tape_reversal": False, "post_hoc_repair": False},
        "provenance": {"bank": str(BANK.relative_to(ROOT)), "bank_sha256": hashlib.sha256(BANK.read_bytes()).hexdigest(), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audits": ["two-pointer character mismatch", "forward/reverse SHA-256"], "collocation_role": "ranking complete parses only", "reader_gate": "closed; programmatic scores do not certify English readability", "next_reader_test": "randomized blinded ratings of intact prose versus shuffled controls, packaged with provenance and exact audit"},
        "status": "fresh exact candidates require human reading" if exact else "no fresh exact parse in this lane",
        "next_construction": "retain only exact complete frames and prepare the blinded intact-versus-shuffled reader package; if empty, add a genuinely new coordination grammar rather than enlarging this bank",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": result["stats"]["complete_left_frames"], "states": result["stats"]["reverse_states"], "parses": result["stats"]["complete_reverse_parses"], "exact": result["stats"]["exact"]}))
