"""Audit saved pair diversity and tag admission for the September 10 critique.

No search rerun or model calls. The tag checker uses position bitsets over the
frozen shapes and does not import the production SentencePlan implementation.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.evidence_paths import evidence_path
INPUTS = [
    "data/structural/aggregate.json",
    "inputs/brown.json.gz",
    "inputs/vocab30k.txt",
]


def audit():
    if sys.flags.optimize:
        raise RuntimeError("Run audits without Python optimization; assertions must remain enabled.")
    aggregate = json.loads(evidence_path(INPUTS[0]).read_text())
    payload = json.loads(gzip.decompress(evidence_path(INPUTS[1]).read_bytes()))
    vocabulary = set(evidence_path(INPUTS[2]).read_text().split()[:30000])
    openings = {"PRON", "DET", "NOUN", "ADJ", "NUM", "ADV"}
    shapes = sorted({tuple(shape) for shape in payload["shapes"]
                     if 3 <= len(shape) <= 9 and shape[0] in openings
                     and "VERB" in shape})
    positions = defaultdict(int)
    by_length = defaultdict(int)
    for index, shape in enumerate(shapes):
        bit = 1 << index
        by_length[len(shape)] |= bit
        for position, tag in enumerate(shape):
            positions[len(shape), position, tag] |= bit

    @lru_cache(maxsize=None)
    def word_mask(length, position, word):
        mask = 0
        for tag in payload["table"].get(word, []):
            mask |= positions[length, position, tag]
        return mask

    def admitted(words):
        mask = by_length[len(words)]
        for position, word in enumerate(words):
            mask &= word_mask(len(words), position, word)
            if not mask:
                return False
        return bool(mask)

    report = {
        "scope": "Saved-output audit, not new search trials or a readability evaluation.",
        "inputs_sha256": {name: hashlib.sha256(evidence_path(name).read_bytes()).hexdigest()
                          for name in INPUTS},
        "retained_shapes": len(shapes),
        "family_definition": "(last left token, first right token); a descriptive junction grouping, not independent samples",
        "arms": {},
    }
    arm_sets = {}
    for name, row in aggregate.items():
        keys, strings = set(), set()
        families, lengths = Counter(), Counter()
        for pair in row["pairs"]:
            left, right = pair["left"].split(), pair["right"].split()
            words = left + right
            assert all(re.fullmatch("[a-z]+", word) and word in vocabulary for word in words)
            assert len(set(words)) == len(words)
            assert admitted(left) and admitted(right)
            assert "".join(left) == "".join(right)[::-1]
            letters = "".join(words)
            assert 20 <= len(letters) <= 44
            keys.add((pair["left"], pair["right"]))
            strings.add(letters)
            families[left[-1], right[0]] += 1
            lengths[len(letters)] += 1
        assert len(keys) == len(row["pairs"]) == row["hits"]
        arm_sets[name] = keys
        report["arms"][name] = {
            "verified_pairs": len(keys),
            "normalized_strings": len(strings),
            "junction_families": len(families),
            "largest_families": [{"junction": list(key), "pairs": count}
                                 for key, count in families.most_common(5)],
            "lengths": dict(sorted(lengths.items())),
            "fraction_at_44_letters": lengths[44] / len(keys),
        }
    report["shared_pairs"] = len(arm_sets["terminal"] & arm_sets["planned"])
    report["exclusive_pairs"] = {}
    for name, other in (("planned", "terminal"), ("terminal", "planned")):
        exclusive = arm_sets[name] - arm_sets[other]
        report["exclusive_pairs"][name] = {
            "pairs": len(exclusive),
            "junction_families": len({(left.split()[-1], right.split()[0]) for left, right in exclusive}),
        }
    report["pair_yield_ratio"] = len(arm_sets["planned"]) / len(arm_sets["terminal"])
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(audit(), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)
