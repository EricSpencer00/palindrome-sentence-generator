"""Freeze a bounded test of sentence boundaries versus exact-pair boundaries.

The v3 hierarchy asks both halves of every exact mirror pair to look like a
sentence.  That is a structural convenience, not a linguistic necessity: a
grammatical sentence may cross the point where an exact pair turns.  This
experiment isolates that choice before changing the service.

It forms two equally sized, seeded two-pair material arms from generated v3
inventory:

* ``sentence_shaped``: pairs that pass the existing POS-shape gate;
* ``cross_boundary_only``: otherwise eligible pairs that fail that gate.

Each block retains both exact outer-pair orders.  The blocks are not selected
for an automatic language score.  A later, separately recorded model screen
may insert punctuation into the *whole* chosen word run, but may never alter
letters or words.  That screen is a development triage tool only; this script
does not claim a reader outcome or modify `/api/v3`.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.first_composition_rescue import (  # noqa: E402
    MirrorPair,
    compose,
    load_pairs,
)
from llm_palindrome.validator import is_palindrome, normalize  # noqa: E402


BLOCKS_PER_ARM = 6


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_pairs(bank_path: Path) -> tuple[list[MirrorPair], list[MirrorPair]]:
    """Return disjoint POS-shaped and boundary-crossing-only inventories."""
    all_pairs = load_pairs(bank_path, require_sentence_pairs=False)
    shaped_ids = {
        pair.pair_id for pair in load_pairs(bank_path, require_sentence_pairs=True)
    }
    sentence_shaped = [pair for pair in all_pairs if pair.pair_id in shaped_ids]
    cross_boundary_only = [pair for pair in all_pairs if pair.pair_id not in shaped_ids]
    if set(pair.pair_id for pair in sentence_shaped).intersection(
            pair.pair_id for pair in cross_boundary_only):
        raise AssertionError("source arms overlap")
    if len(sentence_shaped) + len(cross_boundary_only) != len(all_pairs):
        raise AssertionError("source arms do not cover inventory")
    return sentence_shaped, cross_boundary_only


def sample_blocks(pairs: Sequence[MirrorPair], *, seed: int,
                  blocks: int = BLOCKS_PER_ARM) -> list[tuple[MirrorPair, MirrorPair]]:
    if blocks < 1:
        raise ValueError("blocks must be positive")
    want = blocks * 2
    if len(pairs) < want:
        raise ValueError(f"need {want} eligible pairs, found {len(pairs)}")
    selected = list(pairs)
    random.Random(seed).shuffle(selected)
    selected = selected[:want]
    if len({pair.pair_id for pair in selected}) != want:
        raise AssertionError("sample contains duplicate pairs")
    return list(zip(selected[::2], selected[1::2]))


def record_variant(arm: str, block_id: str, condition: str,
                   outer: MirrorPair, inner: MirrorPair) -> dict:
    words, layout = compose(outer, inner)
    plain = " ".join(words)
    return {
        "source_arm": arm,
        "block_id": block_id,
        "condition": condition,
        "outer_pair_id": outer.pair_id,
        "inner_pair_id": inner.pair_id,
        "plain": plain,
        "letters": len(normalize(plain)),
        "words": len(words),
        "character_multiset": sorted(normalize(plain)),
        "word_multiset": sorted(words),
        "depth": 2,
        "layout": layout,
    }


def assert_blocks(variants: Iterable[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in variants:
        if not is_palindrome(row["plain"]):
            raise AssertionError("candidate is not an exact palindrome")
        grouped.setdefault((row["source_arm"], row["block_id"]), []).append(row)
    for key, rows in grouped.items():
        if len(rows) != 2 or {row["condition"] for row in rows} != {"a", "b"}:
            raise AssertionError(f"{key} does not have both counter-orders")
        for field in ("letters", "words", "character_multiset", "word_multiset", "depth"):
            if rows[0][field] != rows[1][field]:
                raise AssertionError(f"{key} differs on {field}")
        if rows[0]["plain"] == rows[1]["plain"]:
            raise AssertionError(f"{key} has indistinguishable orders")


def build_materials(bank_path: Path, *, seed: int,
                    blocks: int = BLOCKS_PER_ARM) -> dict:
    sentence_shaped, cross_boundary_only = partition_pairs(bank_path)
    arms = {
        "sentence_shaped": sample_blocks(sentence_shaped, seed=seed, blocks=blocks),
        "cross_boundary_only": sample_blocks(cross_boundary_only, seed=seed + 1,
                                             blocks=blocks),
    }
    variants: list[dict] = []
    sampled: dict[str, list[MirrorPair]] = {}
    for arm, arm_blocks in arms.items():
        sampled[arm] = [pair for block in arm_blocks for pair in block]
        for index, (pair_a, pair_b) in enumerate(arm_blocks, 1):
            block_id = f"{arm[:2].upper()}{index:02d}"
            variants.extend((
                record_variant(arm, block_id, "a", pair_a, pair_b),
                record_variant(arm, block_id, "b", pair_b, pair_a),
            ))
    assert_blocks(variants)
    return {
        "status": "frozen_materials_pending_development_screen",
        "design": {
            "seed": seed,
            "blocks_per_arm": blocks,
            "conditions_per_block": ["a", "b"],
            "selection": "seeded uniform sample within each source arm; no language-proxy postselection",
            "display_rule": "any later renderer sees the complete word run and must preserve all letters and words",
        },
        "input_sha256": {
            "data/v3_bank.json": sha256_file(bank_path),
            "experiments/first_composition_rescue.py": sha256_file(
                ROOT / "experiments" / "first_composition_rescue.py"),
            "llm_palindrome/hierarchy.py": sha256_file(ROOT / "llm_palindrome" / "hierarchy.py"),
        },
        "inventory": {
            "all_eligible_pairs": len(sentence_shaped) + len(cross_boundary_only),
            "sentence_shaped_pairs": len(sentence_shaped),
            "cross_boundary_only_pairs": len(cross_boundary_only),
        },
        "sampled_pairs": {arm: [asdict(pair) for pair in pairs]
                          for arm, pairs in sampled.items()},
        "variants": variants,
    }


def write_materials(out_dir: Path, materials: dict) -> None:
    raise RuntimeError(
        "retired: this legacy blind-material writer uses prohibited pair-bank output and "
        "does not implement the required reader controls"
    )


def _retired_write_materials(out_dir: Path, materials: dict) -> None:
    internal = out_dir / "internal"
    blind = out_dir / "blind-screen"
    internal.mkdir(parents=True)
    blind.mkdir()
    (internal / "materials.json").write_text(json.dumps(materials, indent=2) + "\n")
    # The blind view deliberately omits source-arm and counter-order labels.
    items = [{"block_id": row["block_id"], "alternative": row["condition"],
              "text": row["plain"]} for row in materials["variants"]]
    random.Random(materials["design"]["seed"] + 2).shuffle(items)
    (blind / "candidate-word-runs.json").write_text(json.dumps(items, indent=2) + "\n")
    manifest = {str(path.relative_to(out_dir)): sha256_file(path)
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    (out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    raise RuntimeError("retired: invalid legacy blind-material builder")


def _retired_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=ROOT / "data" / "v3_bank.json")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--blocks-per-arm", type=int, default=BLOCKS_PER_ARM)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"output path already exists: {args.out_dir}")
    materials = build_materials(args.bank, seed=args.seed, blocks=args.blocks_per_arm)
    write_materials(args.out_dir, materials)
    print(json.dumps({"out_dir": str(args.out_dir), **materials["inventory"],
                      "blocks": args.blocks_per_arm * 2,
                      "variants": len(materials["variants"])}, indent=2))


if __name__ == "__main__":
    main()
