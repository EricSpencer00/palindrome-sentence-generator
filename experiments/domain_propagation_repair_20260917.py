"""Offset-aware lexical-domain propagation for finite palindrome frames.

This is an implementation repair to the assumption-core slot solver.  It is
deliberately conservative: character support is an over-approximation, so a
surviving value is not a solution.  A value is removed only when no possible
total length, offset, and mirrored-character support can keep it in an exact
palindrome.  The final answer still requires exhaustive assignment and an
independent tape audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _length_sums(domains: Sequence[Sequence[str]]) -> set[int]:
    sums = {0}
    for domain in domains:
        sums = {n + len(value) for n in sums for value in domain}
    return sums


def _prefix_sums(domains: Sequence[Sequence[str]], stop: int) -> set[int]:
    sums = {0}
    for domain in domains[:stop]:
        sums = {n + len(value) for n in sums for value in domain}
    return sums


def _suffix_sums(domains: Sequence[Sequence[str]], start: int) -> set[int]:
    sums = {0}
    for domain in domains[start:]:
        sums = {n + len(value) for n in sums for value in domain}
    return sums


def _offsets_for_value(
    domains: Sequence[Sequence[str]], index: int, value: str, total: int
) -> set[int]:
    """All starts for ``value`` that are compatible with ``total``."""
    prefix = _prefix_sums(domains, index)
    suffix = _suffix_sums(domains, index + 1)
    needed = total - len(value)
    return {start for start in prefix if needed - start in suffix}


def _position_support(domains: Sequence[Sequence[str]], total: int) -> list[set[str]]:
    """Character support at every tape position for one possible total length."""
    support = [set() for _ in range(total)]
    for index, domain in enumerate(domains):
        for value in domain:
            for start in _offsets_for_value(domains, index, value, total):
                for offset, char in enumerate(value):
                    if 0 <= start + offset < total:
                        support[start + offset].add(char)
    return support


def _value_has_support(
    domains: Sequence[Sequence[str]], index: int, value: str, total: int,
    support: Sequence[set[str]],
) -> bool:
    for start in _offsets_for_value(domains, index, value, total):
        if all(
            value[offset] in support[total - 1 - (start + offset)]
            for offset in range(len(value))
        ):
            return True
    return False


def propagate(
    domains: Sequence[Sequence[str]], max_rounds: int = 100
) -> tuple[list[list[str]], dict]:
    """Conservatively prune unsupported lexical values.

    The support sets intentionally ignore cross-slot correlations.  Therefore
    every value used by an exact assignment survives; only a later exhaustive
    pass can certify a palindrome.
    """
    current = [list(dict.fromkeys(domain)) for domain in domains]
    history: list[dict] = []
    for round_no in range(1, max_rounds + 1):
        lengths = sorted(_length_sums(current)) if all(current) else []
        supports = {total: _position_support(current, total) for total in lengths}
        changed = False
        removed: dict[int, list[str]] = {}
        for index, domain in enumerate(current):
            keep = [
                value
                for value in domain
                if any(
                    _value_has_support(current, index, value, total, supports[total])
                    for total in lengths
                )
            ]
            if len(keep) != len(domain):
                changed = True
                removed[index] = [value for value in domain if value not in keep]
                current[index] = keep
        history.append({"round": round_no, "removed": removed, "lengths": lengths})
        if not changed:
            break
        if not all(current):
            break
    return current, {"rounds": history, "fixed_point": not any(history[-1]["removed"].values()) if history else True}


def exhaustive(domains: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    return [
        choice
        for choice in itertools.product(*domains)
        if (joined := "".join(choice)) == joined[::-1]
    ]


def soundness_check() -> dict:
    """Compare propagation against exhaustive solutions on tiny domains."""
    cases = [
        [["a", "ab"], ["a", "ba"]],
        [["ab", "a"], ["ba", "a"]],
        [["a", "b"], ["", "a"], ["a", "b"]],
        [["ab", "c", "ba"], ["ba", "c", "ab"]],
    ]
    checks = []
    for domains in cases:
        solutions = exhaustive(domains)
        reduced, info = propagate(domains)
        reduced_set = {choice for choice in itertools.product(*reduced)} if all(reduced) else set()
        solution_set = set(solutions)
        checks.append({
            "domains": domains,
            "solutions": [list(x) for x in solutions],
            "reduced": reduced,
            "sound": all(any(value in domain for domain in reduced) for choice in solutions for value, domain in zip(choice, reduced)),
            "solution_survives": solution_set <= reduced_set,
            "rounds": info["rounds"],
        })
    return {"cases": checks, "all_solution_sets_survive": all(x["solution_survives"] for x in checks)}


def scene_report() -> dict:
    path = ROOT / "experiments" / "assumption_core_scene_solver_20260916.py"
    spec = importlib.util.spec_from_file_location("assumption_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    scenes = []
    for scene in module.SCENES:
        before = [list(domain) for domain in scene["slots"]]
        reduced, info = propagate(before)
        scenes.append({
            "id": scene["id"],
            "complete_assignments_before": __import__("math").prod(map(len, before)),
            "domain_sizes_before": list(map(len, before)),
            "domain_sizes_after": list(map(len, reduced)),
            "empty_domain": any(not domain for domain in reduced),
            "propagation": info,
            "first_chars": sorted({value[0] for value in before[0]}),
            "last_chars": sorted({value[-1] for value in before[-1]}),
        })
    return {"scenes": scenes, "soundness": soundness_check()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "experiment": "domain-propagation-repair-20260917",
        "status": "implementation_repair_diagnostic",
        "signature": "offset-aware-character-support|conservative-domain-pruning|independent-exhaustive-soundness-check",
        "novelty": {
            "classification": "repair",
            "parent": "assumption-core-scene-solver-20260916",
            "new_family": False,
        },
        "report": scene_report(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_count": 0,
        "mechanically_admitted": 0,
        "reader_eligible": 0,
        "next_repair": "Use the same propagator on fresh optional-constituent domains; retain exhaustive closure and reader review as separate gates.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "scenes": len(report["report"]["scenes"]), "sound": report["report"]["soundness"]["all_solution_sets_survive"]}))


if __name__ == "__main__":
    main()
