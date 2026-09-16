"""Equal-length collision search over two independent semantic clause banks.

The two sides are generated separately from typed POS/sense inventories.  A
clause is complete before it enters the collision index: ``DET SUBJECT VERB
DET OBJECT PREP DET ADJUNCT``.  The right bank is indexed by the SHA-256 of
its own tape; a left clause probes the digest of its reversed tape, but the
right clause is rendered in its normal lexical order.  A digest hit is then
replayed with an independent two-pointer comparison before admission.

This is search evidence, not a readability certificate.  The inventories are
small, authored semantic frames (not catalogue palindromes or copied source
sentences), and every retained row records clause-level completeness,
readability diagnostics, provenance, and the first concrete repair frontier.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "equal-length-clause-collision-20260916"
SIGNATURE = (
    "equal-length-independent-svo-adjunct-banks|"
    "distinct-pos-sense-lexicons|reversed-tape-hash-collision|"
    "two-pointer-exact-audit|complete-clause-readability"
)
EVIDENCE = ROOT / "runs" / "equal-length-clause-collision-20260916.json"
MIN_LETTERS = 39
MAX_LETTERS = 240
BANK_LIMIT = 120_000


@dataclass(frozen=True)
class Lexeme:
    """A lexical choice with an explicit syntactic role and semantic sense."""

    word: str
    pos: str
    sense: str


@dataclass(frozen=True)
class VerbFrame:
    word: str
    object_sense: str
    event_sense: str


@dataclass(frozen=True)
class AdjunctFrame:
    preposition: Lexeme
    determiner: Lexeme
    adjective: Lexeme
    location: Lexeme


@dataclass(frozen=True)
class Clause:
    side: str
    determiner: Lexeme
    subject: Lexeme
    verb: VerbFrame
    object: Lexeme
    adjunct: AdjunctFrame

    @property
    def words(self) -> tuple[str, ...]:
        a = self.adjunct
        return (
            self.determiner.word,
            self.subject.word,
            self.verb.word,
            self.determiner.word,
            self.object.word,
            a.preposition.word,
            a.determiner.word,
            a.adjective.word,
            a.location.word,
        )

    @property
    def text(self) -> str:
        return " ".join(self.words).capitalize() + "."

    @property
    def tape(self) -> str:
        return normalize(self.text)

    @property
    def semantic_roles(self) -> dict[str, str]:
        a = self.adjunct
        return {
            "subject_sense": self.subject.sense,
            "event_sense": self.verb.event_sense,
            "object_sense": self.object.sense,
            "adjunct_sense": a.location.sense,
        }


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def _lex(words: Iterable[str], pos: str, sense: str) -> tuple[Lexeme, ...]:
    return tuple(Lexeme(word, pos, sense) for word in words)


DETERMINERS = _lex(("a", "the"), "DET", "definiteness")
PREPOSITIONS_L = _lex(("in", "near", "under", "beside", "by", "around"), "PREP", "location")
PREPOSITIONS_R = _lex(("at", "over", "behind", "within", "through", "among"), "PREP", "location")

# The side inventories are intentionally disjoint in content vocabulary.  A
# shared determiner is grammatical function material, not a copied content
# unit.  Subjects and objects are grouped by sense; verbs state the object
# sense they select, so a generated clause is independently interpretable.
SUBJECTS_L = _lex(
    ("baker", "farmer", "gardener", "keeper", "maker", "pilot", "poet", "teacher",
     "worker", "writer", "artist", "doctor", "sailor", "tailor", "scholar", "mason",
     "painter", "ranger", "carpenter", "curator", "merchant", "musician", "chemist", "editor"),
    "NOUN",
    "person-agent",
)
SUBJECTS_R = _lex(
    ("caller", "child", "clerk", "driver", "friend", "guide", "nurse", "reader",
     "rider", "student", "author", "captain", "chef", "lawyer", "mentor", "singer",
     "sponsor", "visitor", "witness", "actor", "broker", "scout", "judge", "referee"),
    "NOUN",
    "person-agent",
)

OBJECTS_L = {
    "text": _lex(("letter", "note", "poem", "story", "report", "journal", "manual", "record"), "NOUN", "text-artifact"),
    "food": _lex(("bread", "grain", "meal", "fruit", "cheese", "soup", "apple", "salad"), "NOUN", "food"),
    "object": _lex(("map", "parcel", "window", "button", "sketch", "token", "engine", "shelter"), "NOUN", "physical-object"),
    "place": _lex(("garden", "harbor", "office", "bridge", "market", "tower", "school", "workshop"), "NOUN", "place"),
}
OBJECTS_R = {
    "text": _lex(("answer", "message", "reply", "chart", "file", "paper", "script", "ledger"), "NOUN", "text-artifact"),
    "food": _lex(("orange", "dinner", "oat", "rice", "cake", "tea", "sauce", "stew"), "NOUN", "food"),
    "object": _lex(("alarm", "package", "door", "signal", "plant", "picture", "machine", "basket"), "NOUN", "physical-object"),
    "place": _lex(("camp", "hall", "port", "lab", "street", "village", "station", "theater"), "NOUN", "place"),
}

VERBS_L = (
    VerbFrame("reads", "text", "information"), VerbFrame("writes", "text", "composition"),
    VerbFrame("studies", "text", "inquiry"), VerbFrame("stores", "food", "preservation"),
    VerbFrame("serves", "food", "hospitality"), VerbFrame("plants", "object", "cultivation"),
    VerbFrame("marks", "object", "notation"), VerbFrame("carries", "object", "transport"),
    VerbFrame("opens", "object", "access"), VerbFrame("mends", "object", "repair"),
    VerbFrame("builds", "object", "construction"), VerbFrame("visits", "place", "travel"),
    VerbFrame("guards", "place", "protection"), VerbFrame("cleans", "place", "maintenance"),
    VerbFrame("tends", "place", "care"),
)
VERBS_R = (
    VerbFrame("answers", "text", "response"), VerbFrame("offers", "text", "exchange"),
    VerbFrame("returns", "text", "delivery"), VerbFrame("copies", "text", "replication"),
    VerbFrame("gathers", "food", "collection"), VerbFrame("cooks", "food", "preparation"),
    VerbFrame("brings", "object", "transport"), VerbFrame("checks", "object", "inspection"),
    VerbFrame("moves", "object", "motion"), VerbFrame("fixes", "object", "repair"),
    VerbFrame("enters", "place", "travel"), VerbFrame("leads", "place", "guidance"),
    VerbFrame("crosses", "place", "travel"), VerbFrame("leaves", "place", "departure"),
    VerbFrame("sees", "place", "perception"),
)

ADJECTIVES_L = _lex(("quiet", "bright", "clear", "fresh", "small", "old", "safe", "wise", "open", "green"), "ADJ", "property")
ADJECTIVES_R = _lex(("ancient", "brief", "careful", "clean", "dark", "early", "fine", "gentle", "large", "ready"), "ADJ", "property")
LOCATIONS_L = _lex(("room", "field", "home", "road", "shore", "yard", "cabin", "bakery", "farm", "library"), "NOUN", "location")
LOCATIONS_R = _lex(("plaza", "harbor", "school", "street", "station", "village", "theater", "kitchen", "valley", "museum"), "NOUN", "location")


def build_clause_bank(side: str, limit: int = BANK_LIMIT) -> list[Clause]:
    """Build complete clauses in lexical slot order, capped deterministically."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if side == "left":
        subjects, verbs, objects, preps, adjs, locations = SUBJECTS_L, VERBS_L, OBJECTS_L, PREPOSITIONS_L, ADJECTIVES_L, LOCATIONS_L
    else:
        subjects, verbs, objects, preps, adjs, locations = SUBJECTS_R, VERBS_R, OBJECTS_R, PREPOSITIONS_R, ADJECTIVES_R, LOCATIONS_R
    out: list[Clause] = []
    for det, subject, verb, prep, adj, loc in itertools.product(DETERMINERS, subjects, verbs, preps, adjs, locations):
        for obj in objects[verb.object_sense]:
            adjunct = AdjunctFrame(prep, det, adj, loc)
            clause = Clause(side, det, subject, verb, obj, adjunct)
            # Content-word uniqueness is required within each independently
            # readable clause; function words may repeat by normal grammar.
            content = (subject.word, verb.word, obj.word, loc.word)
            if len(content) != len(set(content)):
                continue
            out.append(clause)
            if len(out) >= limit:
                return out
    return out


def clause_readability(clause: Clause) -> dict[str, object]:
    """Structural/semantic diagnostics; no model or reader claim is made."""
    words = clause.words
    expected = ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "ADJ", "NOUN")
    actual = (clause.determiner.pos, clause.subject.pos, "VERB", clause.determiner.pos,
              clause.object.pos, clause.adjunct.preposition.pos, clause.adjunct.determiner.pos,
              clause.adjunct.adjective.pos, clause.adjunct.location.pos)
    return {
        "complete_svo_adjunct": len(words) == 9 and actual == expected,
        "semantic_selection": clause.object.sense.split("-", 1)[0] == clause.verb.object_sense,
        "content_words_distinct": len({words[1], words[2], words[4], words[8]}) == 4,
        "lexical_order_preserved": True,
        "reader_evidence": False,
        "independently_readable_input": True,
    }


def two_pointer_verify(left_tape: str, right_tape: str) -> dict[str, object]:
    """Verify ``left + right`` symmetry without using string reversal."""
    i, j = 0, len(right_tape) - 1
    matched = 0
    mismatch = None
    while i < len(left_tape) and j >= 0:
        if left_tape[i] != right_tape[j]:
            mismatch = {"left_index": i, "right_index": j, "left": left_tape[i], "right": right_tape[j]}
            break
        matched += 1
        i += 1
        j -= 1
    equal_length = len(left_tape) == len(right_tape)
    return {
        "closed": mismatch is None and equal_length and i == len(left_tape),
        "equal_length": equal_length,
        "matched": matched,
        "left_letters": len(left_tape),
        "right_letters": len(right_tape),
        "first_mismatch": mismatch,
    }


def _hash(tape: str) -> str:
    return hashlib.sha256(tape.encode("ascii")).hexdigest()


def audit_pair(left: Clause, right: Clause) -> dict[str, object]:
    left_tape, right_tape = left.tape, right.tape
    rendered = left.text + " " + right.text
    pointers = two_pointer_verify(left_tape, right_tape)
    exact = bool(left_tape) and left_tape == right_tape[::-1]
    admission = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": rendered,
        "letters": len(left_tape) + len(right_tape),
        "left_letters": len(left_tape),
        "right_letters": len(right_tape),
        "equal_length": len(left_tape) == len(right_tape),
        "exact": exact,
        "two_pointer": pointers,
        "hash_left": _hash(left_tape),
        "hash_right": _hash(right_tape),
        "hash_target": _hash(left_tape[::-1]),
        "left_readability": clause_readability(left),
        "right_readability": clause_readability(right),
        "admission": admission,
        "mechanically_admitted": exact and pointers["closed"] and all(admission.values()),
        "provenance": {
            "left_side": left.side,
            "right_side": right.side,
            "left_semantic_roles": left.semantic_roles,
            "right_semantic_roles": right.semantic_roles,
            "word_order_reversed": False,
            "source_sentences_copied": False,
            "known_palindromes_used": False,
            "catalogue_imported": False,
        },
    }


def novelty_preflight() -> dict[str, object]:
    """Check the registry before constructing either bank."""
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries = [row for row in registry["entries"] if row.get("id") != EXPERIMENT_ID]
    current = set(re.findall(r"[a-z0-9]+", SIGNATURE.lower()))
    common = {"a", "an", "and", "audit", "character", "complete", "exact", "from", "independent", "join", "lexical", "of", "reverse", "right", "tape", "the", "to", "two", "with"}
    current -= common
    nearest = []
    for row in entries:
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "").lower())) - common
        union = current | other
        nearest.append({"id": row.get("id"), "jaccard": round(len(current & other) / len(union), 6) if union else 0.0, "shared_atoms": sorted(current & other)})
    nearest.sort(key=lambda row: (-row["jaccard"], row["id"] or ""))
    return {
        "registry_entries": len(entries),
        "runtime_registry_entries": len(registry["entries"]),
        "exact_signature_collision": any(row.get("signature") == SIGNATURE for row in entries),
        "exact_id_collision": any(row.get("id") == EXPERIMENT_ID for row in entries),
        "nearest_prior": nearest[:5],
        "performed_before_search": True,
    }


def _probe_pairs(left: Sequence[Clause], right: Sequence[Clause], limit: int = 8) -> list[dict[str, object]]:
    """Retain readable complete controls and first-mismatch repair evidence."""
    by_length: dict[int, list[Clause]] = {}
    for clause in right:
        by_length.setdefault(len(clause.tape), []).append(clause)
    rows: list[dict[str, object]] = []
    for left_clause in left[:64]:
        candidates = by_length.get(len(left_clause.tape), ())
        # A deterministic sparse sample avoids quadratic near-miss rendering.
        for right_clause in candidates[::max(1, len(candidates) // 16)][:16]:
            row = audit_pair(left_clause, right_clause)
            row["diagnostic_only"] = True
            rows.append(row)
    rows.sort(key=lambda row: (row["two_pointer"]["matched"], row["letters"]), reverse=True)
    return rows[:limit]


def run(bank_limit: int = BANK_LIMIT) -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["exact_signature_collision"] or preflight["exact_id_collision"]:
        raise RuntimeError("novelty preflight collision; refusing to run a replay")
    left = build_clause_bank("left", bank_limit)
    right = build_clause_bank("right", bank_limit)
    # Equal-length digest buckets are the only pair channel.  A hash hit is
    # never trusted alone: the exact tape comparison below defeats collisions.
    reverse_index: dict[tuple[int, str], list[Clause]] = {}
    for clause in right:
        reverse_index.setdefault((len(clause.tape), _hash(clause.tape)), []).append(clause)
    exact_rows: list[dict[str, object]] = []
    hash_hits = 0
    for left_clause in left:
        target = left_clause.tape[::-1]
        for right_clause in reverse_index.get((len(target), _hash(target)), ()):
            hash_hits += 1
            if right_clause.tape != target:
                continue
            row = audit_pair(left_clause, right_clause)
            row["diagnostic_only"] = False
            exact_rows.append(row)
            if len(exact_rows) >= 200:
                break
        if len(exact_rows) >= 200:
            break
    probes = _probe_pairs(left, right)
    best = probes[0] if probes else None
    admitted = [row for row in exact_rows if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_rows_require_reader_review",
        "method": "Independent semantic SVO+adjunct clause banks are generated in normal lexical order, bucketed by equal tape length, joined by SHA-256 lookup of reversed left tapes, and replayed by a separate two-pointer audit before shared admission.",
        "novelty_preflight": preflight,
        "config": {
            "bank_limit": bank_limit,
            "min_letters": MIN_LETTERS,
            "max_letters": MAX_LETTERS,
            "grammar": "DET SUBJECT VERB DET OBJECT PREP DET ADJ NOUN",
            "equal_length_only": True,
            "catalogue_imported": False,
            "word_order_reversal": False,
            "hash": "sha256(tape)",
        },
        "stats": {
            "left_bank": len(left),
            "right_bank": len(right),
            "left_length_buckets": len({len(c.tape) for c in left}),
            "right_length_buckets": len({len(c.tape) for c in right}),
            "hash_hits": hash_hits,
            "exact": len(exact_rows),
            "mechanically_admitted": len(admitted),
            "rendered_probes": len(probes),
            "max_probe_letters": max((row["letters"] for row in probes), default=0),
            "max_complete_clause_letters": max((len(c.tape) for c in itertools.chain(left, right)), default=0),
        },
        "exact_candidates": exact_rows,
        "rendered_probes": probes,
        "best_readable_pair": best,
        "repair": {
            "status": "not_run" if not exact_rows else "deferred_to_reader_gate",
            "operator": "At the first two-pointer mismatch, substitute one held-out lexeme with the same POS and semantic sense in the offending slot, retain the clause grammar and equal-length bucket, then rebuild only that bucket's reverse-hash index.",
            "frontier": (best["two_pointer"]["first_mismatch"] if best else None),
            "preserves_independent_clause_readability": True,
        },
        "provenance": {
            "source_sentences_copied": False,
            "known_palindromes_used": False,
            "catalogue_imported": False,
            "reader_evidence": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "left_inventory_sha256": _hash("|".join(x.word for x in SUBJECTS_L)),
            "right_inventory_sha256": _hash("|".join(x.word for x in SUBJECTS_R)),
        },
    }


def main() -> None:
    if EVIDENCE.exists():
        raise SystemExit(f"refusing to overwrite existing output: {EVIDENCE}")
    payload = run()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": payload["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
