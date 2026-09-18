"""Freeze a controlled first-composition rescue screen.

The current v3 endpoint turns a POS-shaped mirror half into a sentence and
nests many such pairs.  That combines two untested assumptions: that the
constituent material reads, and that a forced pair boundary is a sentence
boundary.  This experiment separates them without promoting a new API mode.

It samples 12 generated, nondegenerate mirror pairs from the same bank used by
v3's hierarchical mode.  Each pair appears once alone and once in a two-pair
block.  Every block has exactly two possible outer-pair orders.  They contain
the same words, character multiset, normalized-letter count, and depth; only
their order differs.  The renderer sees each full word run once, so it may put
punctuation across the artificial mirror-pair boundary.

This freezes candidate material and both counter-orders.  A later, separately
versioned human- or model-guidance decision may designate one order as guided,
but this script never calls a language model or treats a proxy as readability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.hierarchy import is_sentence_pair
from llm_palindrome.present import present
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import harvest_pair


PAIR_COUNT = 12


@dataclass(frozen=True)
class MirrorPair:
    pair_id: str
    bank_index: int
    origin: str
    source_text_sha256: str
    left: tuple[str, ...]
    right: tuple[str, ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def pair_id(left: Sequence[str], right: Sequence[str]) -> str:
    payload = "\0".join((" ".join(left), " ".join(right))).encode()
    return sha256_bytes(payload)[:16]


def pair_key(left: Sequence[str], right: Sequence[str]) -> tuple[str, str]:
    return normalize(" ".join(left)), normalize(" ".join(right))


def load_pairs(bank_path: Path, *, require_sentence_pairs: bool = True) -> list[MirrorPair]:
    """Generated, splittable, nondegenerate pairs, deduplicated like v3."""
    bank = json.loads(bank_path.read_text())
    table = shapes = trigrams = None
    if require_sentence_pairs:
        table, shapes, trigrams = brown_tables()
    seen_halves: set[str] = set()
    out: list[MirrorPair] = []
    for index, row in enumerate(bank):
        if row.get("source") != "generated":
            continue
        words = row["text"].split()
        got = harvest_pair(words)
        if not got:
            continue
        left, right = got
        left_key, right_key = pair_key(left, right)
        if not left_key or left_key == right_key:
            continue
        if left_key in seen_halves or right_key in seen_halves:
            continue
        if normalize(" ".join(right)) != left_key[::-1]:
            raise AssertionError("bank pair is not a letter mirror")
        if require_sentence_pairs and not is_sentence_pair(left, right, table, shapes, trigrams):
            continue
        seen_halves.update((left_key, right_key))
        out.append(MirrorPair(
            pair_id=pair_id(left, right),
            bank_index=index,
            origin=str(row.get("origin", "")),
            source_text_sha256=sha256_bytes(row["text"].encode()),
            left=tuple(left),
            right=tuple(right),
        ))
    return out


def sample_blocks(pairs: Sequence[MirrorPair], seed: int,
                  pair_count: int = PAIR_COUNT) -> list[tuple[MirrorPair, MirrorPair]]:
    if pair_count <= 0 or pair_count % 2:
        raise ValueError("pair_count must be a positive even number")
    if len(pairs) < pair_count:
        raise ValueError(f"need {pair_count} eligible pairs, found {len(pairs)}")
    chosen = list(pairs)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:pair_count]
    if len({pair.pair_id for pair in chosen}) != pair_count:
        raise AssertionError("sample contains a duplicate pair")
    return list(zip(chosen[::2], chosen[1::2]))


def compose(outer: MirrorPair, inner: MirrorPair) -> tuple[list[str], list[dict]]:
    """``L_outer L_inner R_inner R_outer`` with no centre."""
    words = list(outer.left + inner.left + inner.right + outer.right)
    layout = [
        {"pair_id": outer.pair_id, "role": "outer_left", "words": list(outer.left)},
        {"pair_id": inner.pair_id, "role": "inner_left", "words": list(inner.left)},
        {"pair_id": inner.pair_id, "role": "inner_right", "words": list(inner.right)},
        {"pair_id": outer.pair_id, "role": "outer_right", "words": list(outer.right)},
    ]
    plain = " ".join(words)
    if not is_palindrome(plain):
        raise AssertionError("two-pair composition is not a palindrome")
    return words, layout


def whole_render(words: Sequence[str]) -> str:
    """Presentation gets the whole text, never one call per structural half."""
    table, shapes, trigrams = brown_tables()
    rendered = present(words, table, shapes, trigrams)
    if normalize(rendered) != normalize(" ".join(words)):
        raise AssertionError("renderer changed normalized letters")
    if not is_palindrome(rendered):
        raise AssertionError("renderer broke palindrome exactness")
    return rendered


def record_variant(block_id: str, condition: str, outer: MirrorPair,
                   inner: MirrorPair) -> dict:
    words, layout = compose(outer, inner)
    return {
        "block_id": block_id,
        "condition": condition,
        "outer_pair_id": outer.pair_id,
        "inner_pair_id": inner.pair_id,
        "plain": " ".join(words),
        "rendered": whole_render(words),
        "letters": len(normalize(" ".join(words))),
        "words": len(words),
        "character_multiset": sorted(normalize(" ".join(words))),
        "word_multiset": sorted(words),
        "depth": 2,
        "layout": layout,
    }


def record_constituent(block_id: str, pair: MirrorPair) -> dict:
    words = list(pair.left + pair.right)
    plain = " ".join(words)
    if not is_palindrome(plain):
        raise AssertionError("constituent pair is not a palindrome")
    return {
        "block_id": block_id,
        "condition": "constituent",
        "pair_id": pair.pair_id,
        "plain": plain,
        "rendered": whole_render(words),
        "letters": len(normalize(plain)),
        "words": len(words),
        "depth": 1,
        "layout": [
            {"pair_id": pair.pair_id, "role": "left", "words": list(pair.left)},
            {"pair_id": pair.pair_id, "role": "right", "words": list(pair.right)},
        ],
    }


def assert_counterbalance(variants: Iterable[dict]) -> None:
    for variant in variants:
        if variant["condition"] not in {"order_a_outer", "order_b_outer"}:
            continue
        if variant["depth"] != 2 or not is_palindrome(variant["rendered"]):
            raise AssertionError("invalid two-pair variant")
    by_block: dict[str, list[dict]] = {}
    for variant in variants:
        if variant["condition"] in {"order_a_outer", "order_b_outer"}:
            by_block.setdefault(variant["block_id"], []).append(variant)
    for block_id, rows in by_block.items():
        if len(rows) != 2:
            raise AssertionError(f"{block_id} does not have both orders")
        first, second = rows
        for field in ("letters", "words", "character_multiset", "word_multiset", "depth"):
            if first[field] != second[field]:
                raise AssertionError(f"{block_id} differs on {field}")
        if first["plain"] == second["plain"]:
            raise AssertionError(f"{block_id} counter-orders are identical")


def write_materials(out_dir: Path, blocks: Sequence[tuple[MirrorPair, MirrorPair]],
                    seed: int, bank_path: Path) -> dict:
    raise RuntimeError(
        "retired: this legacy rater package uses prohibited pair/refrain material and "
        "does not implement the required intact-prose and shuffled controls"
    )


def _retired_write_materials(out_dir: Path, blocks: Sequence[tuple[MirrorPair, MirrorPair]],
                            seed: int, bank_path: Path) -> dict:
    pairs = [pair for block in blocks for pair in block]
    variants: list[dict] = []
    for index, (pair_a, pair_b) in enumerate(blocks, 1):
        block_id = f"B{index:02d}"
        variants.extend((
            record_constituent(block_id, pair_a),
            record_constituent(block_id, pair_b),
            record_variant(block_id, "order_a_outer", pair_a, pair_b),
            record_variant(block_id, "order_b_outer", pair_b, pair_a),
        ))
    assert_counterbalance(variants)
    rng = random.Random(seed + 1)
    shuffled = list(variants)
    rng.shuffle(shuffled)
    for index, variant in enumerate(shuffled, 1):
        variant["id"] = f"C{index:03d}"

    internal = out_dir / "internal"
    rater = out_dir / "rater-packet"
    internal.mkdir(parents=True)
    rater.mkdir()
    materials = {
        "status": "frozen_materials_pending_guidance_and_human_evaluation",
        "design": {
            "seed": seed,
            "pair_count": len(pairs),
            "blocks": len(blocks),
            "conditions_per_block": ["constituent_a", "constituent_b", "order_a_outer", "order_b_outer"],
            "guide_status": "Neither order is labelled guided until a separately versioned decision is attached.",
            "renderer": "one llm_palindrome.present() call on each whole word run",
        },
        "input_sha256": {
            "data/v3_bank.json": sha256_file(bank_path),
            "server/v3.py": sha256_file(ROOT / "server" / "v3.py"),
            "llm_palindrome/hierarchy.py": sha256_file(ROOT / "llm_palindrome" / "hierarchy.py"),
            "llm_palindrome/present.py": sha256_file(ROOT / "llm_palindrome" / "present.py"),
            "llm_palindrome/syntax.py": sha256_file(ROOT / "llm_palindrome" / "syntax.py"),
        },
        "pairs": [asdict(pair) for pair in pairs],
        "variants": shuffled,
    }
    (internal / "materials.json").write_text(json.dumps(materials, indent=2) + "\n")
    # The distribution directory excludes sources, bank positions, and condition labels.
    blind = [{"id": row["id"], "text": row["rendered"]} for row in shuffled]
    (rater / "blind-items.json").write_text(json.dumps(blind, indent=2) + "\n")
    instructions = """# First-composition pilot (blinded material)

Read each passage independently. For each, first write a short description of
what you think it means or intends. Then score grammaticality, identifiable
subject or intent, and whole-text coherence from 0 (absent) to 3 (clear).

Do not infer a source or reward a passage merely for unusual form. The passages
are a development screen; your ratings are not a claim about an algorithm.
"""
    (rater / "HUMAN-INSTRUCTIONS.md").write_text(instructions)
    manifest = {str(path.relative_to(out_dir)): sha256_file(path)
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    (out_dir / "MANIFEST-SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return materials


def main() -> None:
    raise RuntimeError("retired: invalid legacy reader-package builder")


def _retired_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--bank", type=Path, default=ROOT / "data" / "v3_bank.json")
    parser.add_argument("--allow-nonsentence-pairs", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"output path already exists: {args.out_dir}")
    pairs = load_pairs(args.bank, require_sentence_pairs=not args.allow_nonsentence_pairs)
    blocks = sample_blocks(pairs, args.seed)
    materials = write_materials(args.out_dir, blocks, args.seed, args.bank)
    print(json.dumps({"out_dir": str(args.out_dir), "eligible_pairs": len(pairs),
                      "blocks": len(blocks), "items": len(materials["variants"])}, indent=2))


if __name__ == "__main__":
    main()
