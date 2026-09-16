"""Same-family repair: richer morphology-first dependency trees.

This repair expands the original probe with independently annotated
derivational families (``teach + -er``, ``kind + -ness``, ``metal + -ic``)
and ordinary declarative, passive, progressive, relative, and imperative
dependency topologies.  Morphology paths are selected before surface yield;
there is no reverse emission, catalogue import, centre-out cursor, or paired
clause product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

FAMILY_ID = "morphology-first-dependency-lattice"
EXPERIMENT_ID = "morphology-first-dependency-lattice-derivational-repair-20260916"
STATE_SPACE_SIGNATURE = (
    "morphology-first-dependency-lattice|lemma-derivation-inflection-path|"
    "typed-dependency-tree-yield|node-local-morphological-unification|"
    "whole-yield-palindrome-audit"
)
ARTIFACT = "experiments/morphology_first_dependency_lattice_derivational_repair_20260916.py"
OUT = ROOT / "runs/morphology-first-dependency-lattice-derivational-repair-20260916.json"
MIN_LETTERS = 15


@dataclass(frozen=True)
class Morph:
    lemma: str
    category: str
    surface: str
    derivation: tuple[str, ...] = ()
    inflection: tuple[str, ...] = ()
    number: str | None = None
    valency: str | None = None


@dataclass(frozen=True)
class Node:
    node_id: str
    relation: str
    category: str
    choices: tuple[Morph, ...]


@dataclass(frozen=True)
class Tree:
    name: str
    nodes: tuple[Node, ...]
    arcs: tuple[tuple[str, str, str], ...]

    def yields(self):
        for values in product(*(node.choices for node in self.nodes)):
            by_id = {node.node_id: value for node, value in zip(self.nodes, values)}
            subj = by_id.get("subj")
            verb = by_id.get("verb") or by_id.get("root")
            if subj and verb and subj.number and verb.number and subj.number != verb.number:
                continue
            if verb and verb.valency == "intransitive" and by_id.get("obj"):
                continue
            yield " ".join(value.surface for value in values), values


def m(lemma: str, category: str, surface: str, *, derivation=(), inflection=(), number=None, valency=None) -> Morph:
    return Morph(lemma, category, surface, tuple(derivation), tuple(inflection), number, valency)


def families() -> dict[str, tuple[Morph, ...]]:
    """Fresh node-local morphology inventory; no catalogue strings are loaded."""
    return {
        "det": tuple(m(x, "DET", x) for x in ("a", "the", "our", "some")),
        "det_sg": tuple(m(x, "DET", x, number="singular") for x in ("a", "the", "our")),
        "det_pl": tuple(m(x, "DET", x, number="plural") for x in ("the", "our", "some")),
        "agent_sg": tuple(m(x, "N", x, number="singular") for x in ("teacher", "writer", "keeper", "maker", "farmer", "artist")) + (
            m("teach", "N", "teacher", derivation=("-er",), number="singular"),
            m("write", "N", "writer", derivation=("-er",), number="singular"),
            m("care", "N", "caretaker", derivation=("-take",), number="singular"),
        ),
        "agent_pl": tuple(m(x, "N", x, inflection=("plural",), number="plural") for x in ("teachers", "writers", "keepers", "makers", "farmers", "artists")),
        "object_sg": tuple(m(x, "N", x, number="singular") for x in ("map", "letter", "plan", "note", "parcel", "garden", "report")),
        "object_pl": tuple(m(x, "N", x, inflection=("plural",), number="plural") for x in ("maps", "letters", "plans", "notes", "parcels", "gardens", "reports")),
        "verb_sg": tuple(m(lemma, "V", surface, inflection=("3sg",), number="singular", valency="transitive") for lemma, surface in (("repair", "repairs"), ("deliver", "delivers"), ("carry", "carries"), ("sort", "sorts"), ("measure", "measures"))),
        "verb_pl": tuple(m(lemma, "V", lemma, number="plural", valency="transitive") for lemma in ("repair", "deliver", "carry", "sort", "measure")),
        "verb_base": tuple(m(lemma, "V", lemma, valency="transitive") for lemma in ("repair", "deliver", "carry", "sort", "measure", "teach", "write")),
        "participle": tuple(m(lemma, "V", surface, derivation=("-ed",), inflection=("participle",), valency="transitive") for lemma, surface in (("repair", "repaired"), ("deliver", "delivered"), ("sort", "sorted"), ("measure", "measured"), ("teach", "taught"))),
        "progressive": tuple(m(lemma, "V", surface, derivation=("-ing",), inflection=("progressive",), valency="transitive") for lemma, surface in (("repair", "repairing"), ("deliver", "delivering"), ("carry", "carrying"), ("measure", "measuring"))),
        "be_sg": (m("be", "AUX", "is", inflection=("present",), number="singular"), m("be", "AUX", "was", inflection=("past",), number="singular")),
        "be_pl": (m("be", "AUX", "are", inflection=("present",), number="plural"), m("be", "AUX", "were", inflection=("past",), number="plural")),
        "adj": (
            m("metal", "ADJ", "metallic", derivation=("-ic",)),
            m("kind", "ADJ", "kind", derivation=("-ness",)),
            m("care", "ADJ", "careful", derivation=("-ful",)),
            m("stress", "ADJ", "stressed", derivation=("-ed",)),
            m("quiet", "ADJ", "quiet"),
            m("clear", "ADJ", "clear"),
        ),
        "rel": (m("that", "REL", "that"), m("who", "REL", "who")),
        "prep": (m("by", "PREP", "by"), m("with", "PREP", "with"), m("near", "PREP", "near")),
    }


def trees() -> tuple[Tree, ...]:
    f = families()
    return (
        Tree("agent_transitive", (Node("det", "det", "DET", f["det_sg"]), Node("subj", "nsubj", "N", f["agent_sg"]), Node("verb", "root", "V", f["verb_sg"]), Node("det2", "det", "DET", f["det_sg"]), Node("obj", "obj", "N", f["object_sg"])), (("verb", "subj", "nsubj"), ("verb", "obj", "obj"), ("subj", "det", "det"), ("obj", "det2", "det"))),
        Tree("plural_transitive", (Node("det", "det", "DET", f["det_pl"]), Node("subj", "nsubj", "N", f["agent_pl"]), Node("verb", "root", "V", f["verb_pl"]), Node("det2", "det", "DET", f["det_pl"]), Node("obj", "obj", "N", f["object_pl"])), (("verb", "subj", "nsubj"), ("verb", "obj", "obj"), ("subj", "det", "det"), ("obj", "det2", "det"))),
        Tree("passive_agent", (Node("det", "det", "DET", f["det_sg"]), Node("subj", "nsubj", "N", f["object_sg"]), Node("root", "root", "AUX", f["be_sg"]), Node("part", "xcomp", "V", f["participle"]), Node("prep", "case", "PREP", (f["prep"][0],)), Node("det2", "det", "DET", f["det_sg"]), Node("obj", "obl", "N", f["agent_sg"])), (("root", "subj", "nsubj"), ("root", "part", "xcomp"), ("part", "obj", "obl"), ("obj", "prep", "case"), ("obj", "det2", "det"))),
        Tree("progressive_event", (Node("det", "det", "DET", f["det_sg"]), Node("subj", "nsubj", "N", f["agent_sg"]), Node("root", "root", "AUX", f["be_sg"]), Node("part", "xcomp", "V", f["progressive"]), Node("det2", "det", "DET", f["det_sg"]), Node("obj", "obj", "N", f["object_sg"])), (("root", "subj", "nsubj"), ("root", "part", "xcomp"), ("part", "obj", "obj"))),
        Tree("relative_modifier", (Node("det", "det", "DET", f["det_sg"]), Node("subj", "nsubj", "N", f["agent_sg"]), Node("rel", "mark", "REL", f["rel"]), Node("verb", "root", "V", f["verb_sg"]), Node("det2", "det", "DET", f["det_sg"]), Node("obj", "obj", "N", f["object_sg"])), (("verb", "subj", "nsubj"), ("subj", "rel", "acl:relcl"), ("verb", "obj", "obj"))),
        Tree("imperative_derivational", (Node("root", "root", "V", f["verb_base"]), Node("det", "det", "DET", f["det_sg"]), Node("obj", "obj", "N", f["object_sg"]), Node("prep", "case", "PREP", (f["prep"][1],)), Node("det2", "det", "DET", f["det_sg"]), Node("obl", "obl", "N", f["agent_sg"])), (("root", "obj", "obj"), ("obj", "prep", "case"), ("obl", "det2", "det"))),
    )


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = [{"left": i, "right": len(tape) - 1 - i, "left_char": tape[i], "right_char": tape[-1 - i]} for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape), "normalized_tape": tape, "mismatch_count": len(mismatches), "mismatches": mismatches[:4], "sha256": hashlib.sha256(tape.encode()).hexdigest()}


def novelty_preflight() -> dict[str, object]:
    rows = json.loads((ROOT / "docs" / "experiment-novelty-registry.json").read_text())["entries"]
    own = [row for row in rows if row["id"] == "morphology-first-dependency-lattice-20260916"]
    foreign_signature = [row["id"] for row in rows if row["signature"] == STATE_SPACE_SIGNATURE and row["id"] not in {x["id"] for x in own}]
    foreign_artifacts = [row["id"] for row in rows if row["id"] not in {x["id"] for x in own} and (row.get("artifact") == ARTIFACT or ARTIFACT in row.get("repair_artifacts", []))]
    return {"registry_entries_read": len(rows), "same_family_base": bool(own), "same_family_signature": [row["id"] for row in own], "foreign_signature_collision": foreign_signature, "foreign_artifact_collision": foreign_artifacts, "status": "same_family_repair" if own and not foreign_signature and not foreign_artifacts else "blocked", "excluded_families": ["center-out", "reverse-segmentation", "grammar-template", "MCTS", "beam", "clause-product"]}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["status"] != "same_family_repair":
        raise RuntimeError(f"same-family preflight failed: {preflight}")
    stats = Counter(tree_count=len(trees()), raw_yields=0, exact_yields=0, catalogue_rejects=0, symmetry_rejects=0, mechanically_admitted=0)
    exact: list[dict[str, object]] = []
    near_misses: list[dict[str, object]] = []
    per_tree: Counter[str] = Counter()
    seen: set[str] = set()
    for tree in trees():
        for text, values in tree.yields():
            stats["raw_yields"] += 1
            per_tree[tree.name] += 1
            checked = audit(text)
            if not checked["exact"] or checked["normalized_tape"] in seen:
                if len(near_misses) < 48 and checked["mismatch_count"]:
                    near_misses.append({"tree": tree.name, "rendered": text.capitalize() + ".", "audit": checked, "morphology_path": [{"lemma": value.lemma, "category": value.category, "surface": value.surface, "derivation": list(value.derivation), "inflection": list(value.inflection)} for value in values]})
                continue
            seen.add(checked["normalized_tape"])
            rendered = text.capitalize() + "."
            checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=220)
            stats["exact_yields"] += 1
            stats["catalogue_rejects"] += int(not checks["absent_from_local_catalogue"])
            stats["symmetry_rejects"] += int(not checks["not_word_order_symmetry"] or not checks["no_self_palindromic_proper_multiword_span"])
            stats["mechanically_admitted"] += int(all(checks.values()))
            exact.append({"tree": tree.name, "rendered": rendered, "audit": checked, "mechanical_checks": checks, "mechanically_admitted": bool(all(checks.values())), "dependency_arcs": [list(arc) for arc in tree.arcs], "morphology_path": [{"lemma": value.lemma, "category": value.category, "surface": value.surface, "derivation": list(value.derivation), "inflection": list(value.inflection)} for value in values], "reader_status": "not_run; exactness and mechanical checks are not readability evidence"})
    exact.sort(key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    near_misses.sort(key=lambda row: (row["audit"]["mismatch_count"], -row["audit"]["letters"], row["rendered"]))
    return {"experiment_id": EXPERIMENT_ID, "family_id": FAMILY_ID, "repair_of": "morphology-first-dependency-lattice-20260916", "state_space_signature": STATE_SPACE_SIGNATURE, "status": "completed", "method": "Expand node-local derivation and inflection paths, enumerate richer valid dependency trees, unify agreement, render ordinary order, and independently audit complete yields.", "novelty_preflight": preflight, "config": {"tree_count": len(trees()), "known_catalogue_imported_for_generation": False, "known_catalogue_used_only_for_admission": True, "reverse_emission": False, "center_out_search": False, "reader_gate_required": True}, "stats": {**dict(stats), "per_tree_yields": dict(per_tree)}, "exact_candidates_and_negative_evidence": exact, "negative_evidence_near_misses": near_misses, "prominent_exact_candidate": exact[0] if exact else None, "next_repair": "If the family is retained, add independently authored derivational paradigms with corpus-backed argument frames and run a blinded intact-vs-shuffled reading study; do not relax catalogue or symmetry gates.", "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "lexical_source": "fresh hand-authored derivation and inflection families", "source_text_copied": False, "readability_certificate": False}, "reader_gate": {"status": "not_run", "eligible": int(stats["mechanically_admitted"]), "reason": "Exactness and morphology do not certify readable prose."}}


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
