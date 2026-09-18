from pathlib import Path

from experiments.cross_boundary_material_probe import (
    assert_blocks,
    build_materials,
    sample_blocks,
)
from experiments.first_composition_rescue import MirrorPair


def pair(index: int) -> MirrorPair:
    left = (f"a{index}",)
    right = (f"b{index}",)
    return MirrorPair(str(index), index, "generated", "hash", left, right)


def test_sample_blocks_is_seeded_and_nonduplicating():
    pool = [pair(index) for index in range(12)]
    first = sample_blocks(pool, seed=7, blocks=4)
    second = sample_blocks(pool, seed=7, blocks=4)
    assert first == second
    assert len({item.pair_id for block in first for item in block}) == 8


def test_assert_blocks_rejects_nonmatching_counterorders():
    good = [
        {"source_arm": "x", "block_id": "X01", "condition": "a", "plain": "step on no pets",
         "letters": 12, "words": 4, "character_multiset": list("ennoppsstteo"),
         "word_multiset": ["no", "on", "pets", "step"], "depth": 2},
        {"source_arm": "x", "block_id": "X01", "condition": "b", "plain": "step on no pets",
         "letters": 12, "words": 4, "character_multiset": list("ennoppsstteo"),
         "word_multiset": ["no", "on", "pets", "step"], "depth": 2},
    ]
    try:
        assert_blocks(good)
    except AssertionError as exc:
        assert "indistinguishable" in str(exc)
    else:
        raise AssertionError("identical counter-orders must fail")


def test_materials_have_disjoint_source_arms_and_exact_variants():
    materials = build_materials(Path("data/v3_bank.json"), seed=13, blocks=1)
    assert materials["inventory"]["sentence_shaped_pairs"] > 0
    assert materials["inventory"]["cross_boundary_only_pairs"] > 0
    left = {p["pair_id"] for p in materials["sampled_pairs"]["sentence_shaped"]}
    right = {p["pair_id"] for p in materials["sampled_pairs"]["cross_boundary_only"]}
    assert left.isdisjoint(right)
    assert len(materials["variants"]) == 4
