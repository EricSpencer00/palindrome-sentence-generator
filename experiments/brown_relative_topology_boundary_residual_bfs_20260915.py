"""Topology/boundary variants for Brown-attested two-relative chains.

This branch changes where a separately attested prepositional attachment is
placed rather than changing relation lexemes.  It tests first-relative suffix,
second-relative prefix, and punctuation-boundary placement, with optional
determiner boundaries.  Tapes used by the previous 497-frame attachment run
are excluded so this is a genuinely new frontier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from experiments.brown_attested_relation_residual_bfs_20260915 import _attested_inflections, _extract_relations
from experiments.brown_shared_coreference_attachment_residual_bfs_20260915 import _attachment_inventory
from experiments.brown_two_relative_chain_residual_bfs_20260915 import _chains as base_chains
from experiments.brown_two_relative_chain_attachment_residual_bfs_20260915 import _chains as attached_chains

MIN_LETTERS = 39
MAX_LETTERS = 300


@dataclass(frozen=True)
class TopologyFrame:
    chain: object
    topology: str
    prep: str
    det: str
    noun: str

    @property
    def pp(self) -> tuple[str, ...]:
        return (self.prep,) + ((self.det,) if self.det else ()) + (self.noun,)

    @property
    def words(self) -> tuple[str, ...]:
        base = list(self.chain.words)
        # Base chain layout: h + who + first verb/det/object + that + second
        # subject/verb.  Positions are found from the fixed token "that" so
        # no hidden text-level splice is possible.
        try:
            that = base.index("that")
        except ValueError:
            return ()
        if self.topology == "first_relative_suffix":
            return tuple(base[:that] + list(self.pp) + base[that:])
        if self.topology == "second_relative_prefix":
            return tuple(base[: that + 1] + list(self.pp) + base[that + 1 :])
        # Cross-boundary placement is after the complete first dependency and
        # before the second dependency's finite verb.
        return tuple(base[: that + 2] + list(self.pp) + base[that + 2 :])

    @property
    def tape(self) -> str:
        return "".join(self.words)

    @property
    def state(self) -> dict:
        return {
            "base_chain_state": self.chain.state,
            "topology": self.topology,
            "attachment": {"prep": self.prep, "det": self.det, "noun": self.noun, "attested": True},
            "boundary_choice": "det_present" if self.det else "det_absent",
            "clause_boundary": "punctuation_only",
        }


def _frames(relations, forms, prepositions, prior_keys, cap=260):
    # The base chain enforces both co-reference edges and agreement.
    bases = tuple(base_chains(relations, forms, cap=cap))
    nouns = tuple(dict.fromkeys(row.object for row in relations[:cap]))
    out = []
    topologies = ("first_relative_suffix", "second_relative_prefix", "cross_boundary")
    for chain in bases:
        for noun in nouns[:24]:
            for prep, det in prepositions.get(noun, ())[:4]:
                for topology in topologies:
                    frame = TopologyFrame(chain, topology, prep, det, noun)
                    if frame.tape not in prior_keys and 22 <= len(frame.tape) <= 150:
                        out.append(frame)
    return tuple(out), len(bases)


def _word_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return tuple(word[::-1] for word in reversed(left)) == right


def _audit(left: TopologyFrame, right: TopologyFrame) -> dict:
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
        "reader_status": "not_run; topology and exactness do not certify readability",
    }


def run() -> dict:
    relations, extraction = _extract_relations(limit=14_000)
    forms = _attested_inflections(relations)
    adjectives, prepositions, attachment_sentences = _attachment_inventory()
    # Exclude every tape from the prior attachment topology run.
    prior = {frame.tape for frame in attached_chains(relations, forms, adjectives, prepositions, cap=len(relations))}
    frames, base_count = _frames(relations, forms, prepositions, prior, cap=len(relations))
    residual: dict[str, list[TopologyFrame]] = defaultdict(list)
    for frame in frames:
        residual[frame.tape].append(frame)
    stats = Counter(extraction)
    stats.update({
        "attachment_brown_sentences_scanned": attachment_sentences,
        "base_two_relative_chains": base_count,
        "prior_497_tape_keys_excluded": len(prior),
        "topology_frames": len(frames),
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
        "status": "complete_relative_topology_boundary_residual_bfs_no_reader_promotion",
        "config": {
            "base_source": "Brown-attested two-relative chains",
            "topologies": ["first_relative_suffix", "second_relative_prefix", "cross_boundary"],
            "attachment_source": "independently Brown-attested preposition-noun pairs",
            "boundary_variants": "determiner present/absent as attested",
            "prior_tape_keys_excluded": True,
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
            "Use a variable-length PP whose noun is co-referent with a chain variable, and search its boundary "
            "position jointly with the reversed residual tape while retaining prior-key exclusion."
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
