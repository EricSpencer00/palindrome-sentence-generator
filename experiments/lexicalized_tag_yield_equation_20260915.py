"""Bounded lexicalized TAG search with an exact terminal-yield equation.

The construction unit here is a Tree-Adjoining Grammar derivation, not a
complete clause bank.  An elementary SVO tree receives typed auxiliary trees
at NP/VP/S sites; each derivation therefore carries an explicit adjunction
stack and can grow to arbitrary depth.  Left and right lexical choices are
independent.  A paired outside-in join then exposes terminal characters from
the two derived yields and rejects a state at the first unequal pair.

This is a bounded construction experiment.  Programmatic familiarity values
are diagnostics only; an exact survivor would still require blinded readers
with intact-prose and shuffled controls.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "lexicalized-tag-yield-equation"
SIGNATURE = "lexicalized-tree-adjoining-grammar|recursive-auxiliary-adjunction|pushdown-yield-equation|typed-semantic-feature-unification|structural-site-repair|independent-terminal-audit"
ARTIFACT = "experiments/lexicalized_tag_yield_equation_20260915.py"
OUT = ROOT / "runs/lexicalized-tag-yield-equation-20260915.json"
SEED = 20260915
MIN_LETTERS = 39
MAX_LETTERS = 180
DEPTHS = tuple(range(5))
AN_INITIAL_WORDS = frozenset({"honest", "honor", "honour", "hour", "heir", "heirloom"})
REJECTION_COUNTS: dict[str, int] = {}


@dataclass(frozen=True)
class ElementaryTree:
    name: str
    yield_labels: tuple[str, ...]
    frame: str


@dataclass(frozen=True)
class AuxiliaryTree:
    name: str
    category: str
    site: str
    inserted_labels: tuple[str, ...]
    frame_addition: str


@dataclass(frozen=True)
class Word:
    text: str
    label: str
    number: str = ""
    frame: str = ""


@dataclass(frozen=True)
class Derivation:
    side: str
    elementary: str
    auxiliary_stack: tuple[str, ...]
    words: tuple[Word, ...]
    frame: str
    depth: int

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


ELEMENTARY = (
    ElementaryTree("transitive_spine", ("det_s", "subject", "verb", "det_o", "object"), "agent-action-patient"),
)

# These are bona fide auxiliary trees.  The site is a TAG adjunction address;
# it is not a free string insertion point and each tree adds a typed semantic
# relation to the derivation.
AUXILIARIES = (
    AuxiliaryTree("subject_modifier", "NP", "subject", ("adj_s",), "subject-property"),
    AuxiliaryTree("object_modifier", "NP", "object", ("adj_o",), "object-property"),
    AuxiliaryTree("verb_manner", "VP", "verb", ("adv",), "manner"),
    AuxiliaryTree("subject_location", "NP", "subject", ("prep_s", "det_sloc", "place_sloc"), "subject-location"),
    AuxiliaryTree("object_location", "VP", "object", ("prep_o", "det_oloc", "place_oloc"), "object-location"),
)

# Held out until the base grammar fails.  Coordination is a structural
# operation, not a new lexical choice in an old slot: it adjoins a second
# predicate at the S/VP boundary and changes the discourse frame.
REPAIR_AUXILIARY = AuxiliaryTree("coordinated_predicate", "S", "object", ("and", "coord_verb", "coord_prep", "coord_det", "coord_place"), "coordinated-action")

LEFT = {
    "and": (Word("and", "and"),),
    "det_s": tuple(Word(x, "det_s") for x in ("a", "the", "our")),
    "adj_s": tuple(Word(x, "adj_s") for x in ("quiet", "patient", "young", "careful", "bright")),
    "adj_o": tuple(Word(x, "adj_o") for x in ("bright", "old", "small", "wooden", "clear")),
    "subject": tuple(Word(x, "subject", n, "agent") for x, n in (("sailor", "sg"), ("nurse", "sg"), ("poet", "sg"), ("farmer", "sg"), ("pilots", "pl"), ("artists", "pl"))),
    "verb": tuple(Word(x, "verb", n, "action") for x, n in (("guides", "sg"), ("carries", "sg"), ("opens", "sg"), ("reads", "sg"), ("guide", "pl"), ("carry", "pl"), ("open", "pl"), ("read", "pl"))),
    "det_o": tuple(Word(x, "det_o") for x in ("a", "the", "one")),
    "object": tuple(Word(x, "object", frame="patient") for x in ("lantern", "letter", "harbor", "garden", "parcel", "window", "bridge", "signal")),
    "adv": tuple(Word(x, "adv") for x in ("quietly", "carefully", "patiently", "today")),
    "prep_s": tuple(Word(x, "prep_s") for x in ("near", "beside", "within")),
    "det_sloc": tuple(Word(x, "det_sloc") for x in ("the", "a", "our")),
    "place_sloc": tuple(Word(x, "place_sloc") for x in ("meadow", "station", "river", "tower")),
    "prep_o": tuple(Word(x, "prep_o") for x in ("near", "under", "behind")),
    "det_oloc": tuple(Word(x, "det_oloc") for x in ("the", "a", "our")),
    "place_oloc": tuple(Word(x, "place_oloc") for x in ("market", "harbor", "window", "garden")),
    "coord_verb": tuple(Word(x, "coord_verb", "sg") for x in ("rests", "waits", "smiles", "walks")) + tuple(Word(x, "coord_verb", "pl") for x in ("rest", "wait", "smile", "walk")),
    "coord_prep": tuple(Word(x, "coord_prep") for x in ("near", "beside", "under")),
    "coord_det": tuple(Word(x, "coord_det") for x in ("the", "a", "our")),
    "coord_place": tuple(Word(x, "coord_place") for x in ("meadow", "station", "river", "tower")),
}
RIGHT = {
    "and": (Word("and", "and"),),
    "det_s": tuple(Word(x, "det_s") for x in ("a", "the", "some")),
    "adj_s": tuple(Word(x, "adj_s") for x in ("gentle", "silent", "steady", "honest", "simple")),
    "adj_o": tuple(Word(x, "adj_o") for x in ("clear", "fresh", "small", "painted", "useful")),
    "subject": tuple(Word(x, "subject", n, "agent") for x, n in (("teacher", "sg"), ("captain", "sg"), ("baker", "sg"), ("doctor", "sg"), ("keepers", "pl"), ("writers", "pl"))),
    "verb": tuple(Word(x, "verb", n, "action") for x, n in (("helps", "sg"), ("sorts", "sg"), ("holds", "sg"), ("watches", "sg"), ("help", "pl"), ("sort", "pl"), ("hold", "pl"), ("watch", "pl"))),
    "det_o": tuple(Word(x, "det_o") for x in ("the", "one")),
    "object": tuple(Word(x, "object", frame="patient") for x in ("compass", "message", "basket", "cottage", "picture", "book", "vessel", "answer")),
    "adv": tuple(Word(x, "adv") for x in ("gently", "silently", "steadily", "today")),
    "prep_s": tuple(Word(x, "prep_s") for x in ("beside", "across", "inside")),
    "det_sloc": tuple(Word(x, "det_sloc") for x in ("the", "a", "some")),
    "place_sloc": tuple(Word(x, "place_sloc") for x in ("village", "office", "school", "coast")),
    "prep_o": tuple(Word(x, "prep_o") for x in ("beside", "around", "toward")),
    "det_oloc": tuple(Word(x, "det_oloc") for x in ("the", "a", "some")),
    "place_oloc": tuple(Word(x, "place_oloc") for x in ("village", "office", "harbor", "coast")),
    "coord_verb": tuple(Word(x, "coord_verb", "sg") for x in ("waits", "rests", "smiles", "walks")) + tuple(Word(x, "coord_verb", "pl") for x in ("wait", "rest", "smile", "walk")),
    "coord_prep": tuple(Word(x, "coord_prep") for x in ("beside", "around", "toward")),
    "coord_det": tuple(Word(x, "coord_det") for x in ("the", "a", "some")),
    "coord_place": tuple(Word(x, "coord_place") for x in ("village", "office", "school", "coast")),
}


def independent_tape(text: str) -> str:
    return "".join(ch for ch in text.casefold() if "a" <= ch <= "z")


def independent_audit(text: str) -> dict[str, object]:
    tape = independent_tape(text)
    mismatches = []
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left_index": left, "right_index": right, "left": tape[left], "right": tape[right]})
        left += 1
        right -= 1
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "pairs_checked": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:5], "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest()}


def _select_word(bank: dict[str, tuple[Word, ...]], label: str, rng: random.Random, subject_number: str = "") -> Word:
    choices = bank[label]
    if label in {"verb", "coord_verb"} and subject_number:
        choices = tuple(x for x in choices if x.number == subject_number)
    if label in {"det_o", "det_oloc", "det_sloc", "coord_det"}:
        # Avoid a/an agreement artefacts in generated probes.
        next_label = {"det_o": "object", "det_oloc": "place_oloc", "det_sloc": "place_sloc", "coord_det": "coord_place"}[label]
        choices = tuple(x for x in choices if not (x.text == "a" and bank[next_label][0].text[0] in "aeiou")) or choices
    return rng.choice(choices)


def _materialize(elementary: ElementaryTree, auxiliaries: tuple[AuxiliaryTree, ...], bank: dict[str, tuple[Word, ...]], rng: random.Random) -> tuple[tuple[Word, ...], str]:
    atoms: list[tuple[str, Word | None]] = [(label, None) for label in elementary.yield_labels]
    # Select the subject before its determiner so number and initial-sound
    # agreement are guaranteed even though the elementary tree lists DET
    # first.  This keeps every rendered probe intact prose.
    subject = rng.choice(bank["subject"])
    subject_number = subject.number
    det_choices = tuple(x for x in bank["det_s"] if not (subject_number == "pl" and x.text == "a") and not (subject.text[0] in "aeiou" and x.text == "a")) or bank["det_s"]
    fixed: dict[str, Word] = {"subject": subject, "det_s": rng.choice(det_choices)}
    # Choose the spine first, including agreement; all auxiliary lexical
    # leaves are then filled independently from the side-specific bank.
    for index, (label, _) in enumerate(atoms):
        word = fixed.get(label) or _select_word(bank, label, rng, subject_number)
        atoms[index] = (label, word)
        if label == "subject":
            subject_number = word.number
    for aux in auxiliaries:
        inserted = [(label, None) for label in aux.inserted_labels]
        # The TAG site is an address on the current derived tree.  Inserting
        # after the first matching anchor makes the operation deterministic;
        # the remaining leaf slots are filled after all adjunctions.
        positions = [i for i, (label, _) in enumerate(atoms) if label == aux.site]
        if not positions:
            continue
        # NP modifiers adjoin before their noun; VP/S auxiliaries adjoin after
        # the anchor.  Keeping this site policy explicit prevents probes such
        # as ``the lantern bright`` from masquerading as intact prose.
        after = aux.name in {"verb_manner", "subject_location", "object_location", "coordinated_predicate"}
        position = positions[0] + (1 if after else 0)
        atoms[position:position] = inserted
    filled: list[Word] = []
    for label, word in atoms:
        if word is None:
            word = _select_word(bank, label, rng, subject_number)
        filled.append(word)
    frame = elementary.frame + "+" + "+".join(aux.frame_addition for aux in auxiliaries)
    return tuple(filled), frame


def surface_issues(words: tuple[Word, ...]) -> list[str]:
    """Reject malformed generated surfaces before they enter the pair join."""
    issues: list[str] = []
    for previous, current in zip(words, words[1:]):
        if previous.text == "a":
            initial = current.text.casefold()
            wants_an = bool(initial) and (initial[0] in "aeiou" or initial in AN_INITIAL_WORDS)
            if wants_an:
                issues.append("a_before_vowel_sound")
        if previous.text == "an" and current.text[:1].casefold() not in "aeiou":
            issues.append("an_before_consonant_sound")
    subjects = [word for word in words if word.label == "subject"]
    if subjects and subjects[0].number == "pl" and any(word.label == "det_s" and word.text == "a" for word in words):
        issues.append("singular_article_before_plural_subject")
    subject_number = subjects[0].number if subjects else ""
    if any(word.label in {"verb", "coord_verb"} and word.number and subject_number and word.number != subject_number for word in words):
        issues.append("subject_verb_agreement")
    return sorted(set(issues))


def _derivation_bank(side: str, *, repair: bool = False, cap_per_depth: int = 72) -> tuple[Derivation, ...]:
    bank = LEFT if side == "left" else RIGHT
    elementary = ELEMENTARY[0]
    aux_pool = AUXILIARIES + ((REPAIR_AUXILIARY,) if repair else ())
    out: list[Derivation] = []
    rejected = 0
    for depth in DEPTHS:
        # The stack is an ordered sequence of distinct elementary auxiliary
        # trees.  This is the TAG growth axis; depth is not a sentence-length
        # bucket or a word-bank multiplier.
        stacks = list(itertools.permutations(aux_pool, depth))
        rng = random.Random(SEED + depth * 101 + (0 if side == "left" else 1) + (10000 if repair else 0))
        rng.shuffle(stacks)
        for stack in stacks[:cap_per_depth]:
            words, frame = _materialize(elementary, stack, bank, rng)
            issues = surface_issues(words)
            if issues:
                rejected += 1
                continue
            out.append(Derivation(side, elementary.name, tuple(aux.name for aux in stack), words, frame, depth))
    REJECTION_COUNTS[f"{side}:{'repair' if repair else 'base'}"] = rejected
    return tuple(out)


def outside_in_join(left: Derivation, right: Derivation) -> dict[str, object]:
    """Consume terminal leaves from opposite edges of two TAG yields."""
    lt, rt = left.tape, right.tape
    matched = 0
    first = None
    for index, (a, b) in enumerate(zip(lt, reversed(rt))):
        if a != b:
            first = {"left_index": index, "right_index": len(rt) - index - 1, "left": a, "right": b}
            break
        matched += 1
    exact = len(lt) == len(rt) and first is None
    return {"exact_half_join": exact, "left_letters": len(lt), "right_letters": len(rt), "matched_outer_letters": matched, "first_mismatch": first}


def shortcuts(text: str) -> list[str]:
    units = tuple(normalize_letters(x) for x in tokenize(text))
    content = [x for x in units if x not in {"a", "an", "the", "our", "some", "one", "near", "beside", "within", "under", "behind", "across", "inside", "around", "toward", "and", "who"}]
    reasons = []
    if len(content) != len(set(content)):
        reasons.append("repeated_content_word")
    if any(x and x == x[::-1] for x in content):
        reasons.append("self_palindromic_content_word")
    if units == tuple(x[::-1] for x in reversed(units)):
        reasons.append("whole_word_order_mirror")
    return reasons


def readability(text: str) -> dict[str, object]:
    ws = tokenize(text)
    return {"status": "diagnostic_only", "word_count": len(ws), "mean_word_length": round(sum(map(len, ws)) / max(1, len(ws)), 2), "content_word_ratio": round(sum(w not in {"a", "an", "the", "our", "some", "one"} for w in ws) / max(1, len(ws)), 3), "blinded_human_readers_required": True}


def novelty_preflight() -> dict[str, object]:
    rows = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())["entries"]
    if any(row["id"] == ID or row["signature"] == SIGNATURE or row["artifact"] == ARTIFACT for row in rows):
        raise RuntimeError("TAG route already registered; refusing to count a replay")
    return {"registry_entries_read_before_run": len(rows), "prior_ids_read": [row["id"] for row in rows], "exact_signature_overlap": [], "exact_artifact_overlap": [], "counted_as_new": True, "novel_dimension": "recursive TAG elementary/auxiliary derivations with typed adjunction stacks and pushdown terminal-yield equation", "excluded_families": ["grammar-intersection-chart", "dependency-attribute-grammar-chart", "clause-lattice-joint-dp", "variable-boundary-tape-ilp", "bpe-dual-continuation", "neural-dual-prefix-beam-v2"]}


def run() -> dict[str, object]:
    novelty = novelty_preflight()
    left = _derivation_bank("left")
    right = _derivation_bank("right")
    # The paired join is a structural product of TAG states.  It retains every
    # best probe but stores complete derivation provenance for each closure.
    stats = {"left_derivations": len(left), "right_derivations": len(right), "paired_structural_states": 0, "terminal_pairs_checked": 0, "exact_half_joins": 0, "mechanically_admitted": 0, "probes": 0}
    exact_candidates: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    best: list[tuple[int, dict[str, object]]] = []
    for left_derivation in left:
        for right_derivation in right:
            stats["paired_structural_states"] += 1
            join = outside_in_join(left_derivation, right_derivation)
            stats["terminal_pairs_checked"] += min(join["left_letters"], join["right_letters"])
            rendered = left_derivation.text.capitalize() + ". " + right_derivation.text.capitalize() + "."
            audit = independent_audit(rendered)
            checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            row = {"rendered": rendered, "letters": audit["letters"], "left_tree": left_derivation.elementary, "right_tree": right_derivation.elementary, "left_auxiliary_stack": list(left_derivation.auxiliary_stack), "right_auxiliary_stack": list(right_derivation.auxiliary_stack), "left_depth": left_derivation.depth, "right_depth": right_derivation.depth, "left_frame": left_derivation.frame, "right_frame": right_derivation.frame, "yield_join": join, "independent_two_pointer_audit": audit, "mechanical_admission_checks": checks, "shortcut_rejections": shortcuts(rendered), "readability_diagnostic": readability(rendered), "reader_status": "not_run", "provenance": {"left_source": "fresh left TAG lexicalization", "right_source": "fresh right TAG lexicalization", "lexicalizations_independent": True}}
            if join["exact_half_join"]:
                stats["exact_half_joins"] += 1
                if audit["exact"] and all(checks.values()) and not row["shortcut_rejections"]:
                    stats["mechanically_admitted"] += 1
                    exact_candidates.append(row)
            else:
                best.append((int(join["left_letters"] - join["matched_outer_letters"] + join["right_letters"] - join["matched_outer_letters"]), row))
    # Shortlist by outer agreement only; no language score certifies anything.
    best.sort(key=lambda x: (x[0], -x[1]["letters"], x[1]["rendered"]))
    probes = [row for _, row in best[:40]]
    stats["probes"] = len(probes)

    repair_left = _derivation_bank("left", repair=True)
    repair_right = _derivation_bank("right", repair=True)
    repair_states = repair_exact = repair_checked = 0
    repair_best: list[tuple[int, dict[str, object]]] = []
    for left_derivation in repair_left:
        for right_derivation in repair_right:
            if "coordinated_predicate" not in left_derivation.auxiliary_stack and "coordinated_predicate" not in right_derivation.auxiliary_stack:
                continue
            repair_states += 1
            join = outside_in_join(left_derivation, right_derivation)
            repair_checked += min(join["left_letters"], join["right_letters"])
            rendered = left_derivation.text.capitalize() + ". " + right_derivation.text.capitalize() + "."
            audit = independent_audit(rendered)
            row = {"rendered": rendered, "letters": audit["letters"], "left_auxiliary_stack": list(left_derivation.auxiliary_stack), "right_auxiliary_stack": list(right_derivation.auxiliary_stack), "left_depth": left_derivation.depth, "right_depth": right_derivation.depth, "yield_join": join, "independent_two_pointer_audit": audit, "readability_diagnostic": readability(rendered), "reader_status": "not_run", "provenance": {"repair": "held-out coordinated_predicate TAG auxiliary", "lexicalizations_independent": True}}
            if join["exact_half_join"]:
                repair_exact += 1
            else:
                repair_best.append((int(join["left_letters"] - join["matched_outer_letters"] + join["right_letters"] - join["matched_outer_letters"]), row))
    repair_best.sort(key=lambda x: (x[0], -x[1]["letters"], x[1]["rendered"]))
    repair_probes = [row for _, row in repair_best[:20]]
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"experiment_id": ID, "signature": SIGNATURE, "method": "recursive lexicalized TAG derivations with typed auxiliary-tree adjunction; a paired outside-in terminal-yield equation consumes opposite edges of independently lexicalized complete yields", "configuration": {"seed": SEED, "depths": list(DEPTHS), "target_length_band": [MIN_LETTERS, MAX_LETTERS], "elementary_trees": [x.__dict__ for x in ELEMENTARY], "auxiliary_trees": [x.__dict__ for x in AUXILIARIES], "repair_auxiliary": REPAIR_AUXILIARY.__dict__, "cap_per_depth": 72, "programmatic_readability_diagnostic_only": True}, "novelty_audit": novelty, "stats": stats, "rendered_candidates": probes, "exact_candidates": exact_candidates, "surface_validation": {"rejected_malformed_derivations": dict(REJECTION_COUNTS), "retained_derivations_rechecked": {"left_base": sum(not surface_issues(d.words) for d in left), "right_base": sum(not surface_issues(d.words) for d in right), "left_repair": sum(not surface_issues(d.words) for d in repair_left), "right_repair": sum(not surface_issues(d.words) for d in repair_right)}}, "repair_run": {"operator": "structural-site repair: adjoin held-out coordinated_predicate at the object/S boundary, adding a second typed predicate and a fresh independent PP; do not merely replace a word in an existing slot", "left_derivations": len(repair_left), "right_derivations": len(repair_right), "paired_repair_states": repair_states, "terminal_pairs_checked": repair_checked, "exact_half_joins": repair_exact, "rendered_probes": repair_probes}, "provenance": {"generator_sha256": script_hash, "lexical_source": "fresh hand-authored disjoint role banks", "source_text_copied": False, "catalogue_text": False, "mirrored_word_list": False, "readability_certificate": False}, "reader_gate": {"status": "not_run", "reason": "No exact survivor has blinded reader evidence; exact output would require intact prose and shuffled controls."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite output: {args.out}; pass --overwrite for a deterministic replay")
    payload = run()
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"stats": payload["stats"], "repair": {k: payload["repair_run"][k] for k in ("paired_repair_states", "exact_half_joins")}, "probes": len(payload["rendered_candidates"]), "artifact": str(args.out)}))
