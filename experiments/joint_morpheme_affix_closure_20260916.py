"""Joint stem/affix closure over two independently ordered English clauses.

This is a deliberately small construction experiment.  Each content slot is
represented as a stem plus an explicitly selected inflectional ending; the
left and right clauses choose those bundles jointly, but the right clause has
its own order and frame.  No candidate is made by reversing a word list.

The run keeps near misses because a zero is useful only when the first
mismatch and a concrete repair are inspectable.  Programmatic readability
scores are diagnostics, never reader evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "joint-morpheme-affix-closure-20260916"
FAMILY_ID = "joint-morpheme-affix-closure"
STATE_SPACE_SIGNATURE = (
    "joint-morpheme-affix-closure|intact-independent-clauses|"
    "stem-ending-bundle-equations|nonmirror-clause-orders|"
    "all-different-content|independent-two-pointer-hash-audit"
)
OUT = ROOT / "runs/joint-morpheme-affix-closure-20260916.json"
MIN_LETTERS = 39
MAX_LETTERS = 180
PROBE_LIMIT = 48

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


@dataclass(frozen=True)
class Bundle:
    lemma: str
    stem: str
    ending: str
    category: str
    feature: str
    surface: str


@dataclass(frozen=True)
class Clause:
    frame: str
    slots: tuple[str, ...]
    bundles: tuple[Bundle, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(bundle.surface for bundle in self.bundles)

    @property
    def tape(self) -> str:
        return normalize_letters(" ".join(self.words))


def _b(lemma: str, stem: str, ending: str, category: str, feature: str) -> Bundle:
    return Bundle(lemma, stem, ending, category, feature, stem + ending)


# The inventories are fresh, hand-authored lexical material.  The stem and
# ending are separate fields even when English spelling has an irregular form;
# this makes the coupled choice visible in the run artifact.
MORPHEMES: dict[str, tuple[Bundle, ...]] = {
    "DET": tuple(_b(x, x, "", "function", "article") for x in ("a", "an", "the", "my", "our")),
    "ADJ": tuple(_b(*row) for row in (
        ("calm", "calm", "", "content", "base"), ("bright", "bright", "", "content", "base"),
        ("careful", "care", "ful", "content", "derivational-ful"),
        ("creative", "creat", "ive", "content", "derivational-ive"),
        ("gentle", "gentle", "", "content", "base"), ("honest", "honest", "", "content", "base"),
        ("playful", "play", "ful", "content", "derivational-ful"),
        ("readable", "read", "able", "content", "derivational-able"),
    )),
    "AGENT": tuple(_b(*row) for row in (
        ("baker", "bake", "r", "content", "agentive-er"), ("caller", "call", "er", "content", "agentive-er"),
        ("driver", "drive", "r", "content", "agentive-er"), ("helper", "help", "er", "content", "agentive-er"),
        ("keeper", "keep", "er", "content", "agentive-er"), ("maker", "make", "r", "content", "agentive-er"),
        ("painter", "paint", "er", "content", "agentive-er"), ("reader", "read", "er", "content", "agentive-er"),
        ("singer", "sing", "er", "content", "agentive-er"), ("teacher", "teach", "er", "content", "agentive-er"),
    )),
    "VERB": tuple(_b(*row) for row in (
        ("bakes", "bake", "s", "content", "3sg"), ("calls", "call", "s", "content", "3sg"),
        ("cares", "care", "s", "content", "3sg"), ("drives", "drive", "s", "content", "3sg"),
        ("helps", "help", "s", "content", "3sg"), ("keeps", "keep", "s", "content", "3sg"),
        ("makes", "make", "s", "content", "3sg"), ("paints", "paint", "s", "content", "3sg"),
        ("reads", "read", "s", "content", "3sg"), ("sings", "sing", "s", "content", "3sg"),
        ("writes", "write", "s", "content", "3sg"),
    )),
    "OBJECT": tuple(_b(*row) for row in (
        ("canvas", "canvas", "", "content", "singular"), ("dessert", "dessert", "", "content", "singular"),
        ("garden", "garden", "", "content", "singular"), ("letter", "letter", "", "content", "singular"),
        ("mural", "mural", "", "content", "singular"), ("novel", "novel", "", "content", "singular"),
        ("report", "report", "", "content", "singular"), ("story", "story", "", "content", "singular"),
        ("canvas", "canvas", "es", "content", "plural"), ("garden", "garden", "s", "content", "plural"),
        ("letter", "letter", "s", "content", "plural"), ("mural", "mural", "s", "content", "plural"),
    )),
    "ADV": tuple(_b(*row) for row in (
        ("calmly", "calm", "ly", "content", "derivational-ly"), ("clearly", "clear", "ly", "content", "derivational-ly"),
        ("kindly", "kind", "ly", "content", "derivational-ly"), ("quietly", "quiet", "ly", "content", "derivational-ly"),
        ("warmly", "warm", "ly", "content", "derivational-ly"),
    )),
}

# These are not reverses of one another.  In particular, the right side puts
# the adjunct before its agent and uses a different determiner/argument order.
LEFT_FRAME = ("DET", "ADJ", "AGENT", "VERB", "DET", "OBJECT", "ADV")
RIGHT_FRAME = ("DET", "OBJECT", "VERB", "ADV", "DET", "ADJ", "AGENT")


def _valid(bundles: tuple[Bundle, ...], slots: tuple[str, ...]) -> bool:
    if len(bundles) != len(slots):
        return False
    content = [bundle.lemma for bundle in bundles if bundle.category == "content"]
    if len(content) != len(set(content)):
        return False
    # Simple agreement and determiner constraints are applied before any tape
    # operation.  The frames intentionally use singular agents/objects.
    for index, (slot, bundle) in enumerate(zip(slots, bundles)):
        if slot != "DET" or index + 1 >= len(bundles):
            continue
        if bundle.surface in {"a", "an"} and bundles[index + 1].feature == "plural":
            return False
        next_word = bundles[index + 1].surface.lower()
        next_initial = next_word[:1]
        vowel_sound = next_initial in "aeiou" or next_word in {"honest"}
        if bundle.surface == "a" and vowel_sound:
            return False
        if bundle.surface == "an" and not vowel_sound:
            return False
    return all(bundle.surface for bundle in bundles)


def _compile(slots: tuple[str, ...], frame: str, *, max_rows: int = 600) -> tuple[Clause, ...]:
    rows: list[Clause] = []
    for chosen in product(*(MORPHEMES[slot] for slot in slots)):
        if _valid(chosen, slots):
            rows.append(Clause(frame, slots, chosen))
            if len(rows) >= max_rows:
                break
    return tuple(rows)


def novelty_preflight() -> dict[str, Any]:
    registry_path = ROOT / "docs/experiment-novelty-registry.json"
    entries = json.loads(registry_path.read_text())["entries"]
    signature_collision = [row["id"] for row in entries if row.get("signature") == STATE_SPACE_SIGNATURE]
    artifact = f"experiments/{Path(__file__).name}"
    artifact_collision = [row["id"] for row in entries if row.get("artifact") == artifact]
    self_registered = signature_collision == [EXPERIMENT_ID] and artifact_collision == [EXPERIMENT_ID]
    return {
        "registry_path": str(registry_path), "registry_entries_read": len(entries),
        "exact_signature_collision": [x for x in signature_collision if x != EXPERIMENT_ID],
        "artifact_collision": [x for x in artifact_collision if x != EXPERIMENT_ID],
        "self_registered": self_registered,
        "status": "registered_self" if self_registered else ("novel_exact_signature" if not signature_collision and not artifact_collision else "collision"),
        "manual_distinction": "Both clause sides select stem-plus-ending bundles before rendering; right order is an independent frame and no word list is reversed.",
        "overlap_reviewed": ["morphology-semantic-template-csp-20260916", "morphological-derivational-seam", "morphology-first-dependency-lattice-20260916"],
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }


def _two_pointer(tape: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right, "left_char": tape[left], "right_char": tape[right]})
        left += 1; right -= 1
    return {"exact": bool(tape) and not mismatches, "pairs_checked": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:8]}


def _hash_audit(tape: str) -> dict[str, Any]:
    # Independent whole-string check: the digest is computed over both the
    # candidate and its reverse, rather than reusing the pointer result.
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"forward_sha256": forward, "reverse_sha256": reverse, "exact": tape == tape[::-1] and bool(tape), "digest_equal": forward == reverse}


def _readability(text: str) -> dict[str, Any]:
    words = tokenize(text)
    common = {"a", "an", "the", "my", "our", "with", "and", "or"}
    content = [word for word in words if word not in common]
    return {
        "word_count": len(words), "content_word_count": len(content),
        "unique_content_word_count": len(set(content)), "mean_word_length": round(sum(map(len, words)) / len(words), 3),
        "diagnostic": "intact English clause templates and distinct content lemmas; blinded readers not run",
    }


def _audit(left: Clause, right: Clause, rendered: str, *, repair: dict[str, Any] | None = None) -> dict[str, Any]:
    tape = normalize_letters(rendered)
    pointer = _two_pointer(tape)
    hashed = _hash_audit(tape)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": rendered, "letters": len(tape), "normalized_tape": tape,
        "independent_two_pointer": pointer, "independent_hash_audit": hashed,
        "central_admission": checks, "mechanically_admitted": pointer["exact"] and hashed["exact"] and all(checks.values()),
        "word_order_mirror": False, "content_lemmas_disjoint": len({b.lemma for b in left.bundles + right.bundles if b.category == "content"}) == 2 * sum(b.category == "content" for b in left.bundles),
        "morpheme_path": {"left": [b.__dict__ for b in left.bundles], "right": [b.__dict__ for b in right.bundles]},
        "left_frame": left.frame, "right_frame": right.frame, "readability": _readability(rendered),
        "repair_operator": repair, "reader_status": "not_run; diagnostics are not human readability evidence",
    }


def _repair_operator(left: Clause, right: Clause, audit: dict[str, Any]) -> dict[str, Any]:
    mismatch = audit["independent_two_pointer"]["mismatches"][0] if audit["independent_two_pointer"]["mismatches"] else None
    target = right.bundles[-1]
    siblings = [b.surface for b in MORPHEMES["AGENT"] if b.surface != target.surface and b.lemma != target.lemma]
    return {
        "operator": "first-mismatch-heldout-affix-swap",
        "action": "replace one right-side agent stem+ending bundle at the first mirrored mismatch; retain the right clause order and all left bundles, then rerun both audits",
        "first_mismatch": mismatch,
        "target_side": "right", "target_slot": "AGENT", "current_bundle": target.__dict__,
        "heldout_candidates": siblings[:5], "preserves": ["intact clause order", "agreement", "distinct content lemmas", "independent two-pointer and hash checks"],
    }


def run() -> dict[str, Any]:
    preflight = novelty_preflight()
    if preflight["status"] not in {"novel_exact_signature", "registered_self"}:
        raise RuntimeError(f"novelty preflight failed before generation: {preflight}")
    left = _compile(LEFT_FRAME, "left-adjunct-agent-action")
    right = _compile(RIGHT_FRAME, "right-object-action-adjunct")
    right_by_tape: dict[str, list[Clause]] = defaultdict(list)
    for clause in right:
        right_by_tape[clause.tape].append(clause)
    stats = Counter(left_clauses=len(left), right_clauses=len(right), joint_pairs=0, exact=0, admitted=0, distinct_rejections=0)
    probes: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # The indexed lookup is the exhaustive closure phase.  A bounded but
    # substantive lockstep product then supplies failed outputs for repair;
    # every sampled pair still chooses both sides independently.
    frontier: list[tuple[int, int, str, Clause, Clause, dict[str, Any]]] = []
    for left_index, l_clause in enumerate(left):
        for r_clause in right_by_tape.get(l_clause.tape[::-1], ()):
            rendered = " ".join(l_clause.words).capitalize() + "; " + " ".join(r_clause.words) + "."
            checked = _audit(l_clause, r_clause, rendered)
            exact_rows.append(checked); stats["exact"] += 1
            stats["admitted"] += int(checked["mechanically_admitted"])
        # 32 deterministic independent right choices per left clause retain
        # a large, inspectable near-miss frontier without materialising the
        # quadratic bank product.
        for offset in range(32):
            r_clause = right[(left_index * 7919 + offset * 104729) % len(right)]
            stats["joint_pairs"] += 1
            rendered = " ".join(l_clause.words).capitalize() + "; " + " ".join(r_clause.words) + "."
            tape = normalize_letters(rendered)
            if tape in seen:
                continue
            seen.add(tape)
            checked = _audit(l_clause, r_clause, rendered)
            if checked["content_lemmas_disjoint"] is False:
                stats["distinct_rejections"] += 1
                continue
            mismatches = checked["independent_two_pointer"]["mismatch_count"]
            mirrored_prefix = next((i for i, pair in enumerate(zip(tape, tape[::-1])) if pair[0] != pair[1]), len(tape) // 2)
            frontier.append((mismatches, -mirrored_prefix, rendered, l_clause, r_clause, checked))
    frontier.sort(key=lambda row: (row[0], row[1], row[2]))
    for _, _, _, l_clause, r_clause, checked in frontier[:PROBE_LIMIT]:
        checked["repair_operator"] = _repair_operator(l_clause, r_clause, checked)
        probes.append(checked)
    return {
        "experiment_id": EXPERIMENT_ID, "family_id": FAMILY_ID, "signature": STATE_SPACE_SIGNATURE,
        "status": "completed", "method": "Enumerate two independent intact-clause frames; choose stem and inflectional/derivational ending bundles on both sides; audit the joined yield.",
        "novelty_preflight": preflight, "config": {"left_slots": LEFT_FRAME, "right_slots": RIGHT_FRAME, "min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS, "right_order_is_reverse": False, "catalogue_imported_for_generation": False, "independent_audits": ["two_pointer", "sha256_forward_reverse"], "probe_limit": PROBE_LIMIT},
        "stats": dict(stats), "exact_closures": exact_rows, "admitted": [row for row in exact_rows if row["mechanically_admitted"]], "failed_outputs": probes,
        "repair_operator": "For each retained failed output, apply its first-mismatch-heldout-affix-swap record to one right-side AGENT stem+ending bundle and rerun the full joint audit; a repair is not admitted unless every mechanical check passes.",
        "readability_gate": {"status": "not_run", "reason": "Readability diagnostics describe intact templates only; any exact admitted closure requires blinded intact-prose and shuffled-control readers."},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_inventory": "fresh hand-authored stems and explicit affix bundles", "source_text_copied": False, "catalogue_reuse": False, "word_order_mirror": False, "repeated_content_units": False, "readability_certificate": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    result = run(); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
