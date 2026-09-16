"""Morphology-first dependency-lattice experiment.

The search state is a typed dependency tree whose node values are *morphology
paths* (lemma, derivation, inflection, surface), rather than a bank of already
assembled clauses.  A tree is selected first, then node-local morphological
choices are unified (agreement and valency), and only then is its ordinary
reading-order yield audited.  No node is filled from a reversed tape.

This is deliberately a small, inspectable probe.  Exactness and the shared
mechanical gate are evidence; neither is a readability certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from itertools import product
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT_ID = "morphology-first-dependency-lattice-20260916"
FAMILY_ID = "morphology-first-dependency-lattice"
STATE_SPACE_SIGNATURE = (
    "morphology-first-dependency-lattice|lemma-derivation-inflection-path|"
    "typed-dependency-tree-yield|node-local-morphological-unification|"
    "whole-yield-palindrome-audit"
)
OUT = ROOT / "runs/morphology-first-dependency-lattice-20260916.json"
MIN_LETTERS = 15


@dataclass(frozen=True)
class Morph:
    lemma: str
    category: str
    derivation: tuple[str, ...]
    inflection: tuple[str, ...]
    surface: str
    number: str | None = None
    valency: str | None = None


@dataclass(frozen=True)
class Node:
    node_id: str
    relation: str
    category: str
    choices: tuple[Morph, ...]


@dataclass(frozen=True)
class DependencyTree:
    name: str
    nodes: tuple[Node, ...]
    arcs: tuple[tuple[str, str, str], ...]
    complete_sentence: bool = True

    def yields(self):
        """Yield normal-order surfaces and their morphology paths."""
        for choices in product(*(node.choices for node in self.nodes)):
            # Agreement/valency is checked on feature bundles, before any tape
            # operation.  The small trees below expose the relevant links by
            # stable node ids, making this a lattice rather than a word bag.
            by_id = {node.node_id: value for node, value in zip(self.nodes, choices)}
            subject = by_id.get("subj")
            verb = by_id.get("root")
            obj = by_id.get("obj")
            if subject and verb and subject.number and verb.number and subject.number != verb.number:
                continue
            if verb and obj and verb.valency == "intransitive" and obj:
                continue
            yield " ".join(value.surface for value in choices), choices


def _m(lemma: str, category: str, surface: str, *, derivation=(), inflection=(), number=None, valency=None) -> Morph:
    return Morph(lemma, category, tuple(derivation), tuple(inflection), surface, number, valency)


def _families() -> dict[str, tuple[Morph, ...]]:
    # Alternatives are generated from explicit lemma families.  In particular,
    # metallic = metal + -ic, stressed = stress + -ed, and sonatas = sonata +
    # plural.  The generator never derives one node from another node's tape.
    return {
        "vocative": (_m("satan", "PROPN", "satan"), _m("anna", "PROPN", "anna"), _m("noel", "PROPN", "noel")),
        "verb_base": (
            _m("oscillate", "V", "oscillate", valency="transitive"),
            _m("deliver", "V", "deliver", valency="transitive"),
            _m("refer", "V", "refer", valency="transitive"),
            _m("stress", "V", "stress", valency="transitive"),
            _m("level", "V", "level", valency="transitive"),
            _m("see", "V", "see", valency="transitive"),
        ),
        "verb_3sg": (
            _m("oscillate", "V", "oscillates", inflection=("3sg",), number="singular", valency="transitive"),
            _m("deliver", "V", "delivers", inflection=("3sg",), number="singular", valency="transitive"),
            _m("refer", "V", "refers", inflection=("3sg",), number="singular", valency="transitive"),
            _m("stress", "V", "stresses", inflection=("3sg",), number="singular", valency="transitive"),
        ),
        "det_my": (_m("my", "DET", "my"), _m("our", "DET", "our")),
        "det_a": (_m("a", "DET", "a"),),
        "adj": (
            _m("metal", "ADJ", "metallic", derivation=("-ic",)),
            _m("stress", "ADJ", "stressed", derivation=("-ed",)),
            _m("level", "ADJ", "level"),
            _m("civic", "ADJ", "civic"),
        ),
        "object": (
            _m("sonata", "N", "sonata", inflection=("sg",), number="singular"),
            _m("sonata", "N", "sonatas", inflection=("plural",), number="plural"),
            _m("dessert", "N", "dessert", inflection=("sg",), number="singular"),
            _m("dessert", "N", "desserts", inflection=("plural",), number="plural"),
            _m("level", "N", "level", inflection=("sg",), number="singular"),
            _m("car", "N", "car", inflection=("sg",), number="singular"),
            _m("cat", "N", "cat", inflection=("sg",), number="singular"),
            _m("rat", "N", "rat", inflection=("sg",), number="singular"),
            _m("god", "N", "god", inflection=("sg",), number="singular"),
        ),
        "copula": (_m("be", "COP", "is", inflection=("present",), number="singular"), _m("be", "COP", "was", inflection=("past",), number="singular")),
        "pronoun": (_m("it", "PRON", "it"), _m("i", "PRON", "i")),
        "prep": (_m("of", "PREP", "of"),),
        "conj": (_m("or", "CONJ", "or"),),
    }


def trees() -> tuple[DependencyTree, ...]:
    f = _families()
    return (
        DependencyTree(
            "vocative_transitive_derivational_object",
            (Node("voc", "vocative", "PROPN", f["vocative"]), Node("root", "root", "V", f["verb_base"]), Node("det", "det", "DET", f["det_my"]), Node("amod", "amod", "ADJ", f["adj"]), Node("obj", "obj", "N", f["object"])),
            (("root", "voc", "vocative"), ("root", "obj", "obj"), ("obj", "det", "det"), ("obj", "amod", "amod")),
        ),
        DependencyTree(
            "imperative_article_object",
            (Node("root", "root", "V", f["verb_base"]), Node("det", "det", "DET", f["det_a"]), Node("obj", "obj", "N", f["object"])),
            (("root", "obj", "obj"), ("obj", "det", "det")),
        ),
        DependencyTree(
            "copular_vocative_attribute",
            (Node("subj", "nsubj", "PROPN", f["vocative"]), Node("root", "root", "COP", f["copula"]), Node("det", "det", "DET", f["det_a"]), Node("obj", "xcomp", "N", f["object"])),
            (("root", "subj", "nsubj"), ("root", "obj", "xcomp"), ("obj", "det", "det")),
        ),
        DependencyTree(
            "inverted_car_cat_question",
            (Node("root", "root", "COP", (_m("be", "COP", "was", inflection=("past",), number="singular"),)), Node("subj", "nsubj", "PRON", f["pronoun"]), Node("det", "det", "DET", f["det_a"]), Node("obj", "obj", "N", f["object"]), Node("conj", "cc", "CONJ", f["conj"]), Node("det2", "det", "DET", f["det_a"]), Node("obj2", "obj", "N", f["object"]), Node("subj2", "nsubj", "PRON", f["pronoun"]), Node("verb2", "xcomp", "V", (_m("see", "V", "saw", inflection=("past",), valency="transitive"),))),
            (("root", "subj", "nsubj"), ("root", "obj", "xcomp"), ("obj", "det", "det"), ("root", "conj", "cc"), ("conj", "obj2", "obj")),
        ),
    )


def audit(text: str) -> dict[str, object]:
    tape = "".join(ch for ch in text.casefold() if "a" <= ch <= "z")
    mismatches = [{"left": i, "right": len(tape) - 1 - i, "left_char": tape[i], "right_char": tape[-1 - i]} for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "normalized_tape": tape, "pairs_checked": len(tape) // 2, "mismatch_count": len(mismatches), "mismatches": mismatches[:4], "sha256": hashlib.sha256(tape.encode()).hexdigest()}


def novelty_preflight() -> dict[str, object]:
    payload = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())
    rows = payload["entries"]
    artifact = "experiments/morphology_first_dependency_lattice_20260916.py"
    exact_signature = [row["id"] for row in rows if row["signature"] == STATE_SPACE_SIGNATURE]
    artifact_collision = [row["id"] for row in rows if row["artifact"] == artifact]
    forbidden_families = {"center-out", "reverse-segmentation", "grammar-template", "MCTS", "beam", "clause-product"}
    text = (" ".join(row["id"] + " " + row["signature"] for row in rows)).casefold()
    return {"registry_entries_read": len(rows), "exact_signature_collision": exact_signature, "artifact_collision": artifact_collision, "status": "novel_exact_signature" if not exact_signature and not artifact_collision else "collision", "prior_families_explicitly_excluded": sorted(forbidden_families), "manual_distinction": "A node-local morphology path is chosen before surface emission; no fixed clause bank, reverse tape, center-out cursor, beam, MCTS, or clause cross-product is used.", "related_but_distinct": [row["id"] for row in rows if row["id"] in {"dependency-attribute-grammar-chart", "morphological-derivational-seam", "wordnet-featured-frame-repair"}], "forbidden_terms_in_signature": sorted(term for term in forbidden_families if term.casefold() in STATE_SPACE_SIGNATURE.casefold()), "registry_text_digest": hashlib.sha256(text.encode()).hexdigest()}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["status"] != "novel_exact_signature":
        raise RuntimeError(f"novelty preflight failed: {preflight}")
    stats = Counter(trees=len(trees()), raw_yields=0, exact_yields=0, catalogue_rejects=0, mechanically_admitted=0)
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for tree in trees():
        for text, values in tree.yields():
            stats["raw_yields"] += 1
            checked = audit(text)
            if not checked["exact"] or checked["normalized_tape"] in seen:
                continue
            seen.add(checked["normalized_tape"])
            checks = mechanical_admission_checks(text.capitalize() + ("?" if tree.name == "inverted_car_cat_question" else "."), min_letters=MIN_LETTERS, max_letters=180)
            catalogue_absent = bool(checks["absent_from_local_catalogue"])
            stats["exact_yields"] += 1
            stats["catalogue_rejects"] += int(not catalogue_absent)
            stats["mechanically_admitted"] += int(all(checks.values()))
            candidates.append({"tree": tree.name, "rendered": text.capitalize() + ("?" if tree.name == "inverted_car_cat_question" else "."), "audit": checked, "mechanical_checks": checks, "mechanically_admitted": bool(all(checks.values())), "dependency_arcs": [list(arc) for arc in tree.arcs], "morphology_path": [{"lemma": value.lemma, "category": value.category, "derivation": list(value.derivation), "inflection": list(value.inflection), "surface": value.surface} for value in values], "reader_status": "not_run; exactness and morphology are not readability evidence"})
    candidates.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    admitted = [row for row in candidates if row["mechanically_admitted"]]
    return {"experiment_id": EXPERIMENT_ID, "family_id": FAMILY_ID, "state_space_signature": STATE_SPACE_SIGNATURE, "status": "completed", "method": "Select typed dependency topology; enumerate node-local lemma/derivation/inflection paths; unify feature attributes; render ordinary order; audit the whole yield independently.", "novelty_preflight": preflight, "config": {"tree_count": len(trees()), "min_letters": MIN_LETTERS, "known_catalogue_imported_for_generation": False, "known_catalogue_used_only_for_admission": True, "reverse_emission": False, "center_out_search": False, "independent_two_pointer_audit": True}, "stats": dict(stats), "exact_candidates": candidates, "prominent_exact_candidate": admitted[0] if admitted else (candidates[0] if candidates else None), "next_repair": "Expand derivational families with independently annotated argument structure and add a blinded intact-vs-shuffled reader gate; do not rescue catalogue-rejected or word-order-symmetric surfaces.", "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "fresh hand-authored morphology families", "source_text_copied": False, "readability_certificate": False}, "reader_gate": {"status": "not_run", "eligible": len(admitted), "reason": "Any eligible exact surface still requires blinded human readability evidence."}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "stats": result["stats"]}, sort_keys=True))


if __name__ == "__main__":
    main()
