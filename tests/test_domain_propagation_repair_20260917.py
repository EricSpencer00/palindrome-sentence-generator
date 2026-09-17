import itertools
import random

from experiments.domain_propagation_repair_20260917 import (
    exhaustive,
    propagate,
    scene_report,
    soundness_check,
)


def test_withheld_tiny_cases_keep_every_exact_solution():
    result = soundness_check()
    assert result["all_solution_sets_survive"] is True


def test_random_tiny_domains_are_conservatively_pruned():
    rng = random.Random(20260917)
    alphabet = "ab"
    for slots in range(1, 5):
        for _ in range(100):
            domains = []
            for _ in range(slots):
                values = {
                    "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 3)))
                    for _ in range(rng.randint(1, 3))
                }
                domains.append(sorted(values))
            solutions = set(exhaustive(domains))
            reduced, _ = propagate(domains)
            retained = set(itertools.product(*reduced)) if all(reduced) else set()
            assert solutions <= retained


def test_inherited_scene_frames_reject_before_complete_rendering():
    report = scene_report()
    assert all(scene["empty_domain"] for scene in report["scenes"])
    assert all(
        scene["complete_assignments_before"] == 139968
        for scene in report["scenes"]
    )
