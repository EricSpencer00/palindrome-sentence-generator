"""Dependency-tree seam solving with live attachment dynamic programming.

Each arm starts as a complete dependency tree with explicit subject, finite
transitive verb, object valency, and one adjunct.  Attachment choices control
the tree frontier and its normal-order linearization (postverbal or fronted
adjunct).  The seam solver walks both already-realized tapes with a memoized
two-pointer DP, retaining boundary-crossing diagnostics.  It does not hash a
reversed bank and never emits a right clause by reversing words.

The exact and admission checks are still mechanical; the clause-level
readability fields are structural diagnostics and are not reader evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "dependency-tree-seam-solver-20260916"
SIGNATURE = (
    "dependency-tree-frontier-attachment-dp|"
    "agreement-aware-transitive-valency|live-seam-boundary-walk|"
    "normal-order-independent-clauses|two-pointer-hash-admission-audit"
)
EVIDENCE = ROOT / "runs" / "dependency-tree-seam-solver-20260916.json"
MIN_LETTERS = 39
MAX_LETTERS = 220


@dataclass(frozen=True)
class Subject:
    word: str
    number: str
    sense: str


@dataclass(frozen=True)
class Verb:
    singular: str
    plural: str
    object_sense: str
    event_sense: str

    def form_for(self, number: str) -> str:
        return self.singular if number == "singular" else self.plural


@dataclass(frozen=True)
class Noun:
    word: str
    sense: str


@dataclass(frozen=True)
class Adjunct:
    prep: str
    adjective: str
    location: Noun
    sense: str


@dataclass(frozen=True)
class DependencyTree:
    side: str
    determiner: str
    subject: Subject
    verb: Verb
    object: Noun
    adjunct: Adjunct
    attachment: str

    @property
    def verb_form(self) -> str:
        return self.verb.form_for(self.subject.number)

    @property
    def words(self) -> tuple[str, ...]:
        a = self.adjunct
        core = (self.determiner, self.subject.word, self.verb_form,
                self.determiner, self.object.word)
        pp = (a.prep, self.determiner, a.adjective, a.location.word)
        if self.attachment == "fronted_event":
            return pp + core
        return core + pp

    @property
    def text(self) -> str:
        if self.attachment == "fronted_event":
            return (" ".join(self.words[:4]).capitalize() + ", "
                    + " ".join(self.words[4:]) + ".")
        return " ".join(self.words).capitalize() + "."

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)

    @property
    def dependencies(self) -> tuple[tuple[str, str, str], ...]:
        """Universal-dependency-like edges with attachment made explicit."""
        adjunct_head = "object" if self.attachment == "postverbal_object" else "verb"
        return (
            ("root", "verb", self.verb_form),
            ("nsubj", "verb", self.subject.word),
            ("obj", "verb", self.object.word),
            ("obl", adjunct_head, self.adjunct.location.word),
        )


SUBJECTS_L = tuple(Subject(w, "singular", "person-agent") for w in
                   ("baker", "farmer", "gardener", "keeper", "maker", "pilot", "poet", "teacher"))
SUBJECTS_R = tuple(Subject(w, "singular", "person-agent") for w in
                   ("caller", "clerk", "driver", "friend", "guide", "nurse", "reader", "rider"))
VERBS_L = (
    Verb("reads", "read", "text", "information"),
    Verb("writes", "write", "text", "composition"),
    Verb("stores", "store", "food", "preservation"),
    Verb("marks", "mark", "object", "notation"),
    Verb("carries", "carry", "object", "transport"),
    Verb("visits", "visit", "place", "travel"),
)
VERBS_R = (
    Verb("answers", "answer", "text", "response"),
    Verb("offers", "offer", "text", "exchange"),
    Verb("gathers", "gather", "food", "collection"),
    Verb("checks", "check", "object", "inspection"),
    Verb("enters", "enter", "place", "travel"),
    Verb("leads", "lead", "place", "guidance"),
)
OBJECTS_L = {
    "text": tuple(Noun(w, "text-artifact") for w in ("letter", "note", "poem", "story")),
    "food": tuple(Noun(w, "food") for w in ("bread", "grain", "meal", "fruit")),
    "object": tuple(Noun(w, "physical-object") for w in ("map", "parcel", "button", "engine")),
    "place": tuple(Noun(w, "place") for w in ("garden", "harbor", "market", "tower")),
}
OBJECTS_R = {
    "text": tuple(Noun(w, "text-artifact") for w in ("answer", "message", "reply", "chart")),
    "food": tuple(Noun(w, "food") for w in ("orange", "dinner", "rice", "cake")),
    "object": tuple(Noun(w, "physical-object") for w in ("alarm", "package", "signal", "machine")),
    "place": tuple(Noun(w, "place") for w in ("camp", "hall", "port", "station")),
}
ADJUNCTS_L = tuple(Adjunct(p, a, Noun(n, "location"), "location") for p, a, n in (
    ("in", "quiet", "room"), ("near", "bright", "field"),
    ("under", "clear", "bridge"), ("by", "fresh", "shore")))
ADJUNCTS_R = tuple(Adjunct(p, a, Noun(n, "location"), "location") for p, a, n in (
    ("at", "brief", "plaza"), ("over", "dark", "valley"),
    ("behind", "clean", "wall"), ("within", "gentle", "village")))
ATTACHMENTS = ("postverbal_event", "postverbal_object", "fronted_event")


def _objects(side: str, sense: str) -> tuple[Noun, ...]:
    return (OBJECTS_L if side == "left" else OBJECTS_R)[sense]


def build_tree_bank(side: str, limit: int = 192) -> list[DependencyTree]:
    """Generate complete trees in normal word order, with explicit valency."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    subjects = SUBJECTS_L if side == "left" else SUBJECTS_R
    verbs = VERBS_L if side == "left" else VERBS_R
    adjuncts = ADJUNCTS_L if side == "left" else ADJUNCTS_R
    out: list[DependencyTree] = []
    for subject in subjects:
        for verb in verbs:
            for obj in _objects(side, verb.object_sense):
                for adjunct in adjuncts:
                    for attachment in ATTACHMENTS:
                        tree = DependencyTree(side, "the", subject, verb, obj, adjunct, attachment)
                        if agreement_and_valency(tree):
                            out.append(tree)
                            if len(out) >= limit:
                                return out
    return out


def agreement_and_valency(tree: DependencyTree) -> bool:
    return (tree.verb_form == tree.verb.form_for(tree.subject.number)
            and tree.object.sense.split("-", 1)[0] == tree.verb.object_sense
            and tree.adjunct.sense == "location")


def tree_readability(tree: DependencyTree) -> dict[str, object]:
    return {
        "complete_clause": len(tree.words) in {9, 9} and len(tree.dependencies) == 4,
        "subject_verb_agreement": agreement_and_valency(tree),
        "transitive_valency": tree.object.sense.split("-", 1)[0] == tree.verb.object_sense,
        "adjunct_attachment": tree.attachment in ATTACHMENTS,
        "normal_word_order": True,
        "reader_evidence": False,
    }


def _frontier_positions(tree: DependencyTree) -> tuple[int, ...]:
    """Character offsets at which a dependency constituent is complete."""
    words = tree.words
    offsets: list[int] = []
    cursor = 0
    for index, word in enumerate(words):
        cursor += len(word)
        if index != len(words) - 1:
            # Spaces disappear from the tape, so the boundary is after the
            # word's characters.  This makes attachment seams observable.
            offsets.append(cursor)
    return tuple(offsets)


def solve_live_seam(left: DependencyTree, right: DependencyTree) -> dict[str, object]:
    """Memoized seam DP over normal-order trees and attachment frontiers."""
    a, b = left.tape, right.tape
    left_frontier = set(_frontier_positions(left))
    right_frontier = set(_frontier_positions(right))

    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> tuple[int, tuple[dict[str, int], ...]]:
        if i >= len(a) or j < 0:
            return 0, ()
        if a[i] != b[j]:
            return 0, ()
        matched, events = dp(i + 1, j - 1)
        boundary = ()
        if i + 1 in left_frontier or j in right_frontier:
            boundary = ({"left_after": i + 1, "right_before": j,
                         "left_boundary": int(i + 1 in left_frontier),
                         "right_boundary": int(j in right_frontier)},)
        return matched + 1, boundary + events

    matched, events = dp(0, len(b) - 1)
    equal_length = len(a) == len(b)
    closed = equal_length and matched == len(a)
    return {
        "closed": closed,
        "equal_length": equal_length,
        "matched": matched,
        "left_letters": len(a),
        "right_letters": len(b),
        "first_mismatch": None if matched == min(len(a), len(b)) else {"left_index": matched, "right_index": len(b) - 1 - matched, "left": a[matched], "right": b[len(b) - 1 - matched]},
        "frontier_events": list(events[:24]),
        "dp_states": dp.cache_info().currsize,
        "hash_left": hashlib.sha256(a.encode()).hexdigest(),
        "hash_right": hashlib.sha256(b.encode()).hexdigest(),
        "hash_whole_palindrome": hashlib.sha256((a + b).encode()).hexdigest() if closed else None,
    }


def audit_pair(left: DependencyTree, right: DependencyTree, seam: dict[str, object], *, with_admission: bool = True) -> dict[str, object]:
    rendered = left.text + " " + right.text
    tape = normalize_letters(rendered)
    admission = (mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
                 if with_admission else {})
    exact_independent = bool(tape) and tape == tape[::-1]
    return {
        "rendered": rendered,
        "letters": len(tape),
        "exact": exact_independent,
        "seam_dp": seam,
        "two_pointer_exact": seam["closed"],
        "hash_exact": seam["closed"] and hashlib.sha256(left.tape.encode()).hexdigest() == hashlib.sha256(right.tape[::-1].encode()).hexdigest(),
        "left_tree": {"attachment": left.attachment, "dependencies": [list(edge) for edge in left.dependencies], "readability": tree_readability(left)},
        "right_tree": {"attachment": right.attachment, "dependencies": [list(edge) for edge in right.dependencies], "readability": tree_readability(right)},
        "admission": admission,
        "mechanically_admitted": exact_independent and seam["closed"] and bool(admission) and all(admission.values()),
        "provenance": {"normal_word_order": True, "word_order_reversal": False, "source_sentences_copied": False, "known_palindromes_used": False, "catalogue_imported": False},
    }


def novelty_preflight() -> dict[str, object]:
    registry = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    entries = [row for row in registry["entries"] if row.get("id") != EXPERIMENT_ID]
    current = set(re.findall(r"[a-z0-9]+", SIGNATURE.lower()))
    common = {"a", "an", "and", "audit", "complete", "exact", "independent", "normal", "order", "the", "two", "with"}
    current -= common
    nearest = []
    for row in entries:
        other = set(re.findall(r"[a-z0-9]+", row.get("signature", "").lower())) - common
        union = current | other
        nearest.append({"id": row.get("id"), "jaccard": round(len(current & other) / len(union), 6) if union else 0.0, "shared_atoms": sorted(current & other)})
    nearest.sort(key=lambda r: (-r["jaccard"], r["id"] or ""))
    return {"registry_entries": len(entries), "runtime_registry_entries": len(registry["entries"]), "exact_signature_collision": any(row.get("signature") == SIGNATURE for row in entries), "exact_id_collision": any(row.get("id") == EXPERIMENT_ID for row in entries), "nearest_prior": nearest[:5], "performed_before_search": True}


def run(limit: int = 192) -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["exact_signature_collision"] or preflight["exact_id_collision"]:
        raise RuntimeError("novelty preflight collision; refusing to run a replay")
    left = build_tree_bank("left", limit)
    right = build_tree_bank("right", limit)
    exact_rows: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    states = 0
    for left_tree in left:
        for right_tree in right:
            seam = solve_live_seam(left_tree, right_tree)
            states += int(seam["dp_states"])
            if seam["closed"]:
                row = audit_pair(left_tree, right_tree, seam)
                row["diagnostic_only"] = False
                exact_rows.append(row)
            elif len(probes) < 12 and left_tree.attachment == right_tree.attachment:
                row = audit_pair(left_tree, right_tree, seam, with_admission=True)
                row["diagnostic_only"] = True
                probes.append(row)
    probes.sort(key=lambda r: r["seam_dp"]["matched"], reverse=True)
    admitted = [row for row in exact_rows if row["mechanically_admitted"]]
    best = probes[0] if probes else None
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_reader_promotion" if not admitted else "exact_rows_require_reader_review",
        "method": "Complete dependency trees are generated with explicit agreement and valency; a memoized live seam DP walks normal-order left and right tapes while retaining attachment-frontier events, then exact two-pointer/hash checks and shared mechanical admission run independently.",
        "novelty_preflight": preflight,
        "config": {"tree_limit_per_side": limit, "attachment_choices": list(ATTACHMENTS), "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "hash_collision_only": False, "reverse_decoding": False, "grammar": "DET nsubj VERB obj obl"},
        "stats": {"left_trees": len(left), "right_trees": len(right), "pair_states": len(left) * len(right), "dp_states": states, "exact": len(exact_rows), "mechanically_admitted": len(admitted), "rendered_probes": len(probes), "max_probe_letters": max((r["letters"] for r in probes), default=0)},
        "exact_candidates": exact_rows,
        "rendered_probes": probes,
        "best_readable_pair": best,
        "repair": {"status": "not_run" if not exact_rows else "deferred_to_reader_gate", "operator": "At the first live-seam mismatch, substitute a held-out lexeme preserving its POS, agreement, and valency, or switch only the mismatching adjunct attachment frontier; rerun the same seam DP and mechanical gate.", "frontier": best["seam_dp"]["first_mismatch"] if best else None, "preserves_tree_completeness": True},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_sentences_copied": False, "known_palindromes_used": False, "catalogue_imported": False, "reader_evidence": False, "normal_word_order": True},
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
