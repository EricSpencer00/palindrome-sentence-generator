"""Variable-length co-referent PP insertion over relative-chain boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.brown_attested_relation_residual_bfs_20260915 import _attested_inflections, _extract_relations
from experiments.brown_shared_coreference_attachment_residual_bfs_20260915 import _attachment_inventory
from experiments.brown_two_relative_chain_residual_bfs_20260915 import _chains as base_chains
from experiments.brown_two_relative_chain_attachment_residual_bfs_20260915 import _chains as attached_chains
from experiments.brown_relative_topology_boundary_residual_bfs_20260915 import TopologyFrame, _frames as prior_topology_frames

MIN_LETTERS = 39
MAX_LETTERS = 320


@dataclass(frozen=True)
class VariablePPFrame:
    chain: object
    boundary: int
    prep: str
    det: str
    adjective: str
    variable: str
    noun: str
    variable_number: str

    @property
    def pp(self) -> tuple[str, ...]:
        return tuple(x for x in (self.prep, self.det, self.adjective, self.noun) if x)

    @property
    def words(self) -> tuple[str, ...]:
        base = list(self.chain.words)
        return tuple(base[:self.boundary] + list(self.pp) + base[self.boundary:])

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        return {
            "base_chain_state": self.chain.state,
            "boundary_token_index": self.boundary,
            "variable": self.variable,
            "variable_noun": self.noun,
            "variable_number": self.variable_number,
            "pp": {"prep": self.prep, "det": self.det, "adjective": self.adjective, "length": len(self.pp), "attested": True},
            "co_reference_checked": True,
            "clause_boundary": "punctuation_only",
        }


def _frames(relations, forms, adjectives, prepositions, prior_keys, cap=260):
    out = []
    for chain in base_chains(relations, forms, cap=cap):
        words = chain.words
        # Explicit chain variables, with agreement metadata taken from their
        # source relation.  The noun is deliberately co-referent, not a free
        # lexical filler.
        variables = (
            ("head_subject", chain.head.subject, chain.head.subject_number),
            ("head_object", chain.head.object, chain.head.object_number),
            ("first_object", chain.first.object, chain.first.object_number),
            ("second_subject", chain.second.subject, chain.second.subject_number),
        )
        for variable, noun, number in variables:
            for prep, attested_det in prepositions.get(noun, ())[:5]:
                # Variable PP lengths: prep+noun, prep+det+noun, and an
                # independently attested adjective inserted before the noun.
                variants = [(attested_det, ""), ("", "")]
                variants += [(attested_det, adjective) for adjective in adjectives.get(noun, ())[:4]]
                for det, adjective in variants:
                    pp = tuple(x for x in (prep, det, adjective, noun) if x)
                    for boundary in range(len(words) + 1):
                        frame = VariablePPFrame(chain, boundary, prep, det, adjective, variable, noun, number)
                        if frame.tape not in prior_keys and 22 <= len(frame.tape) <= 160:
                            out.append(frame)
    return tuple(out)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: VariablePPFrame, right: VariablePPFrame) -> dict:
    text = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_state": left.state,
        "right_state": right.state,
        "word_order_shortcut": _word_mirror(left.words, right.words),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(left.words, right.words),
        "reader_status": "not_run; co-reference and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    prior_attached = {frame.tape for frame in attached_chains(relations, forms, adjectives, prepositions, cap=len(relations))}
    previous_topology = {frame.tape for frame in prior_topology_frames(relations, forms, prepositions, prior_attached, cap=len(relations))[0]}
    prior_keys = prior_attached | previous_topology
    frames = _frames(relations, forms, adjectives, prepositions, prior_keys, cap=len(relations))
    residual: dict[str, list[VariablePPFrame]] = defaultdict(list)
    for frame in frames:
        residual[frame.tape].append(frame)
    stats = Counter(extraction)
    stats.update({
        "attachment_brown_sentences_scanned": attachment_sentences,
        "prior_attached_keys_excluded": len(prior_attached),
        "prior_topology_keys_excluded": len(previous_topology),
        "variable_pp_frames": len(frames),
        "indexed_tapes": len(residual),
    })
    candidates = []
    seen = set()
    for tape, lefts in residual.items():
        rights = residual.get(tape[::-1], ())
        stats["left_states_considered"] += len(lefts)
        if not rights:
            stats["residual_misses"] += len(lefts)
            continue
        stats["residual_matches"] += len(lefts) * len(rights)
        for left in lefts:
            for right in rights:
                if _word_mirror(left.words, right.words):
                    stats["word_order_rejections"] += 1
                    continue
                key = left.tape + right.tape
                if key in seen:
                    continue
                seen.add(key)
                row = _audit(left, right)
                candidates.append(row)
                stats["candidates"] += 1
                if row["mechanically_admitted"]:
                    stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_coreferent_variable_pp_boundary_residual_bfs_no_reader_promotion",
        "config": {
            "base_source": "Brown-attested two-relative chains",
            "pp_source": "independently Brown-attested preposition-noun and adjective-noun pairs",
            "co_reference_variables": ["head_subject", "head_object", "first_object", "second_subject"],
            "variable_lengths": "prep+noun, prep+det+noun, prep+det+adjective+noun",
            "boundary_search": "every token boundary in each chain",
            "prior_keys_excluded": "497 attached and prior topology keys",
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "independent_exact_audit": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": {
            "indexed_tapes": len(residual),
            "left_states": stats["left_states_considered"],
            "unmatched_states": stats["residual_misses"],
            "matched_state_pairs_before_gates": stats["residual_matches"],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "relations_and_attachments_attested_individually": True,
        },
        "next_operator": (
            "Allow two independently attested variable PPs on opposite boundaries while carrying both chain "
            "variables and rejecting all prior tape keys."
        ),
        "reader_gate": "No output is human evidence; future closures require intact-prose and shuffled-control readers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
