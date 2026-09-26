"""Post-hoc, one-factor-at-a-time resource sensitivity follow-up to search v4.

This protocol compares the original-v3 settings with four variants. Each
variant changes one resource setting and holds the other three at v3 values.
It is exploratory: these factors were selected after the adaptive v4 bundle
result was observed. It does not replace or mutate the frozen v4 experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_PATH = ROOT / "experiments/paired_directional_search_resource_sensitivity_20260925.py"
BASE_SPEC = importlib.util.spec_from_file_location("frozen_v4_search", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("could not load the frozen v4 search implementation")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

EXPERIMENT = "paired-directional-search-resource-oat-20260926"
PROTOCOL_VERSION = 1
SEEDS = tuple(range(5000, 5020))
LENGTH_BANDS = ((30, 49), (50, 79), (80, 119))
METHODS = BASE.METHODS
TARGET_PYTHON_VERSION = "3.12.3"
TARGET_PYTHON_IMPLEMENTATION = "CPython"
TARGET_SYSTEM = "Linux"
TARGET_MACHINE = "x86_64"
CONFIGURATION_NAMES = (
    "original_v3",
    "beam_width_256_only",
    "candidate_limit_800_only",
    "per_parent_32_only",
    "budget_180_seconds_only",
)
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_REPLICATES = 20_000
SOURCE_MANIFEST_DEFAULT = ROOT / "runs/directional-search-resource-sensitivity-20260925/manifest.json"
RESULT_DIRECTORY = ROOT / "runs/directional-search-resource-oat-20260926"
REQUIREMENTS_PATH = ROOT / "experiments/resource-oat-20260926-requirements.txt"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pinned_dependencies() -> dict[str, str]:
    dependencies = {}
    for raw_line in REQUIREMENTS_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, pinned = line.partition("==")
        if not separator or not name or not pinned:
            raise ValueError(f"un-pinned runtime dependency: {line}")
        canonical = name.lower()
        if canonical in dependencies:
            raise ValueError(f"duplicate runtime dependency: {name}")
        dependencies[canonical] = pinned
    if not dependencies:
        raise ValueError("runtime requirements must contain unique exact pins")
    return dependencies


def _verify_dependencies(expected: dict[str, str]) -> None:
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise ValueError(f"runtime dependency versions differ from the frozen lock: {observed}")


def _read_source_manifest(path: Path) -> tuple[dict, Path]:
    resolved = path.resolve()
    data = json.loads(resolved.read_text())
    if data.get("experiment") != "paired-directional-search-resource-sensitivity-20260925":
        raise ValueError("source manifest is not the frozen v4 resource-sensitivity manifest")
    BASE.verify_frozen_manifest(data)
    _verify_dependencies(_pinned_dependencies())
    expected_original = {
        "beam_width": 60,
        "candidate_limit": 200,
        "per_parent": 8,
        "max_steps": 400,
        "diversity": 0.4,
        "max_overhang": 24,
        "max_words": 32,
        "cooperative_budget_seconds": 45,
    }
    if data["resource_configurations"].get("original_v3") != expected_original:
        raise ValueError("frozen source does not contain the expected original-v3 settings")
    expected_words = data["vocabulary"]["filtered_word_sha256"]
    words_hash = sha256_bytes(("\n".join(data["vocabulary"]["words"]) + "\n").encode())
    if expected_words != words_hash:
        raise ValueError("frozen v4 vocabulary hash mismatch")
    if sha256_file(BASE_PATH) != data["source_sha256"]["experiment_script"]:
        raise ValueError("frozen v4 runner hash mismatch")
    return data, resolved


def _configurations(source_manifest: dict) -> dict[str, dict[str, Any]]:
    original = dict(source_manifest["resource_configurations"]["original_v3"])
    return {
        "original_v3": original,
        "beam_width_256_only": {**original, "beam_width": 256},
        "candidate_limit_800_only": {**original, "candidate_limit": 800},
        "per_parent_32_only": {**original, "per_parent": 32},
        "budget_180_seconds_only": {**original, "cooperative_budget_seconds": 180},
    }


def _source_inventory(source_manifest: dict) -> dict[str, str]:
    sources = dict(source_manifest["sources"])
    included = set(sources.values())
    for path in sorted((ROOT / "llm_palindrome").glob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative not in included:
            sources[f"runtime_module_{path.stem}"] = relative
            included.add(relative)
    sources["ablation_protocol"] = Path(__file__).resolve().relative_to(ROOT).as_posix()
    sources["runtime_requirements"] = REQUIREMENTS_PATH.relative_to(ROOT).as_posix()
    return sources


def _jobs(source_manifest: dict) -> list[dict[str, Any]]:
    configurations = _configurations(source_manifest)
    names = tuple(configurations)
    jobs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for band_id, (minimum, maximum) in enumerate(LENGTH_BANDS, 1):
            pair_id = f"exploratory:seed-{seed}:band-{band_id}"
            config_order = names if (seed + band_id) % 2 == 0 else tuple(reversed(names))
            execution_order = 0
            for config_position, config_name in enumerate(config_order, 1):
                method_order = (
                    METHODS if (seed + band_id + config_position) % 2 == 0
                    else tuple(reversed(METHODS))
                )
                for method in method_order:
                    execution_order += 1
                    jobs.append({
                        "phase": "exploratory",
                        "pair_id": pair_id,
                        "seed": seed,
                        "band_id": band_id,
                        "min_letters": minimum,
                        "max_letters": maximum,
                        "method": method,
                        "resource_configuration": config_name,
                        "configuration_position": config_position,
                        "execution_order_within_seed_band_block": execution_order,
                    })
    return jobs


def prepare(source_manifest_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output_path}")
    source_manifest, source_path = _read_source_manifest(source_manifest_path)
    if source_path.is_relative_to(ROOT):
        source_path_from_root = source_path.relative_to(ROOT).as_posix()
    else:
        raise ValueError("source manifest must be inside the repository for portable reproduction")

    sources = _source_inventory(source_manifest)
    source_hashes = {name: sha256_file(ROOT / path) for name, path in sources.items()}
    dependencies = _pinned_dependencies()
    _verify_dependencies(dependencies)
    wordfreq_version = importlib.metadata.version("wordfreq")
    expected_wordfreq = source_manifest["provenance"]["wordfreq_required_version"].split()[0]
    if wordfreq_version != expected_wordfreq:
        raise ValueError(f"wordfreq version mismatch: expected {expected_wordfreq}, found {wordfreq_version}")

    manifest = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "status": "post_hoc_exploratory_protocol_frozen_before_new_seed_run",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "motivation": (
            "v4 improved basic-valid coverage from 0/120 to 66/120 while four resource settings changed together; "
            "this follow-up tests each setting separately. Factor selection is post hoc."
        ),
        "design": {
            "primary_question": "What is the marginal effect on basic-valid coverage of changing each v4 resource setting individually from the original-v3 configuration?",
            "unit": "paired search task by seed, length band, and native algorithm",
            "independent_cluster": "seed; all bands, methods, and configurations for a seed stay together in bootstrap resampling",
            "seeds": list(SEEDS),
            "length_bands_normalized_ascii_letters_inclusive": [list(x) for x in LENGTH_BANDS],
            "methods": list(METHODS),
            "primary_endpoint": "four paired differences in basic-valid outputs per requested task; denominator 120 per configuration (20 seeds x 3 bands x 2 methods)",
            "secondary_endpoints": [
                "eligible exact-closure task count, regardless of selected output",
                "strict-admission selected-output count (descriptive only)",
                "per-method basic-valid task counts",
                "deadline exits and elapsed time",
            ],
            "pairing_and_order": "both configurations and methods run for each seed-band block; execution order alternates by frozen seed/band parity",
            "resource_isolation": "five conditions: original v3, plus one separate variant for each of beam_width=256, candidate_limit=800, per_parent=32, or cooperative_budget_seconds=180; all other settings remain at original-v3 values",
            "analysis": f"20,000 deterministic seed-cluster bootstrap replicates with seed {BOOTSTRAP_SEED}; unadjusted and max-absolute simultaneous percentile intervals; descriptive because factor selection follows v4",
            "claim_boundary": "post-hoc one-factor-at-a-time component sensitivity; cannot identify interactions or explain the combined v4 effect by itself; not an independent preregistered confirmation, human readability result, or external-baseline comparison",
        },
        "resource_configurations": _configurations(source_manifest),
        "runtime_dependencies": dependencies,
        "task_count_by_configuration": {name: 120 for name in CONFIGURATION_NAMES},
        "source_manifest": source_path_from_root,
        "source_manifest_sha256": sha256_file(source_path),
        "source_manifest_experiment": source_manifest["experiment"],
        "vocabulary": source_manifest["vocabulary"],
        "sources": sources,
        "source_sha256": source_hashes,
        "citations": source_manifest["citations"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "jobs": _jobs(source_manifest),
        "provenance": {
            "project_git_commit_at_freeze": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "python_version_at_freeze": platform.python_version(),
            "host_at_freeze": platform.node(),
            "required_run_python_version": TARGET_PYTHON_VERSION,
            "required_run_python_implementation": TARGET_PYTHON_IMPLEMENTATION,
            "required_run_platform": f"{TARGET_SYSTEM} {TARGET_MACHINE}",
            "wordfreq_required_version": expected_wordfreq,
            "source_v4_protocol_sha256": source_manifest["source_manifest_sha256"]
            if "source_manifest_sha256" in source_manifest else sha256_file(source_path),
        },
    }
    if len(manifest["jobs"]) != 600:
        raise AssertionError("frozen protocol must contain exactly 600 jobs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def verify_manifest(manifest_path: Path, manifest: dict, *, enforce_target_runtime: bool = True) -> None:
    if manifest.get("experiment") != EXPERIMENT or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected resource OAT protocol")
    source_path = ROOT / manifest["source_manifest"]
    if not source_path.resolve().is_relative_to(ROOT):
        raise ValueError("source manifest path escapes the repository")
    if sha256_file(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("source v4 manifest hash mismatch")
    source_manifest, _ = _read_source_manifest(source_path)
    if manifest.get("resource_configurations") != _configurations(source_manifest):
        raise ValueError("resource settings differ from the validated v4 manifest")
    if manifest.get("vocabulary") != source_manifest.get("vocabulary"):
        raise ValueError("frozen vocabulary differs from the validated v4 manifest")
    expected_sources = _source_inventory(source_manifest)
    expected_hashes = {name: sha256_file(ROOT / path) for name, path in expected_sources.items()}
    if manifest.get("sources") != expected_sources or manifest.get("source_sha256") != expected_hashes:
        raise ValueError("source inventory differs from the frozen v4 inputs plus this protocol")
    for name, relative in manifest["sources"].items():
        if sha256_file(ROOT / relative) != manifest["source_sha256"][name]:
            raise ValueError(f"ablation source hash mismatch: {name}")
    words = manifest["vocabulary"]["words"]
    if sha256_bytes(("\n".join(words) + "\n").encode()) != manifest["vocabulary"]["filtered_word_sha256"]:
        raise ValueError("frozen vocabulary does not match its hash")
    if manifest.get("runtime_dependencies") != _pinned_dependencies():
        raise ValueError("runtime dependency lock differs from the committed requirements")
    _verify_dependencies(manifest["runtime_dependencies"])
    if manifest["provenance"].get("required_run_python_version") != TARGET_PYTHON_VERSION:
        raise ValueError("manifest's target Python version differs from this protocol")
    if manifest["provenance"].get("required_run_python_implementation") != TARGET_PYTHON_IMPLEMENTATION:
        raise ValueError("manifest's target Python implementation differs from this protocol")
    if enforce_target_runtime:
        observed_platform = f"{platform.system()} {platform.machine()}"
        if platform.python_implementation() != TARGET_PYTHON_IMPLEMENTATION:
            raise ValueError(f"runtime requires {TARGET_PYTHON_IMPLEMENTATION}; found {platform.python_implementation()}")
        if platform.python_version() != TARGET_PYTHON_VERSION:
            raise ValueError(f"runtime requires {TARGET_PYTHON_IMPLEMENTATION} {TARGET_PYTHON_VERSION}; found {platform.python_version()}")
        if observed_platform != f"{TARGET_SYSTEM} {TARGET_MACHINE}":
            raise ValueError(f"runtime requires {TARGET_SYSTEM} {TARGET_MACHINE}; found {observed_platform}")
    wordfreq_version = importlib.metadata.version("wordfreq")
    if wordfreq_version != manifest["provenance"]["wordfreq_required_version"]:
        raise ValueError("wordfreq dependency differs from the frozen environment")
    if manifest["jobs"] != _jobs(source_manifest):
        raise ValueError("frozen task rows differ from the seed/configuration generation rule")
    if len(manifest["jobs"]) != 600:
        raise ValueError("frozen job list does not contain 600 tasks")
    counts = {name: 0 for name in CONFIGURATION_NAMES}
    for job in manifest["jobs"]:
        counts[job["resource_configuration"]] += 1
    if counts != {name: 120 for name in CONFIGURATION_NAMES}:
        raise ValueError(f"unexpected job denominators: {counts}")
    if sha256_file(Path(__file__).resolve()) != manifest["source_sha256"]["ablation_protocol"]:
        raise ValueError("running ablation script differs from the frozen protocol")


def run(manifest_path: Path, output_path: Path, workers: int) -> dict:
    if not 1 <= workers <= 6:
        raise ValueError("choose between one and six workers")
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest_path, manifest)
    BASE.RESOURCE_CONFIGURATIONS = manifest["resource_configurations"]
    BASE.EXPERIMENT = EXPERIMENT
    groups: dict[str, list[dict]] = {}
    for job in manifest["jobs"]:
        groups.setdefault(job["pair_id"], []).append(job)
    expected = {(config, method) for config in CONFIGURATION_NAMES for method in METHODS}
    pair_jobs = []
    for group in groups.values():
        observed = {(job["resource_configuration"], job["method"]) for job in group}
        if len(group) != 10 or observed != expected:
            raise ValueError(f"incomplete paired block: {group[0]['pair_id']}")
        group.sort(key=lambda job: job["execution_order_within_seed_band_block"])
        pair_jobs.append(group)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=BASE._worker_init,
        initargs=(str(manifest_path),),
    ) as pool:
        paired_rows = list(pool.map(BASE._worker_run_pair, pair_jobs, chunksize=1))
    rows = list(itertools.chain.from_iterable(paired_rows))
    if len(rows) != 600:
        raise AssertionError(f"expected 600 rows, got {len(rows)}")
    valid_counts = {
        config: sum(row["resource_configuration"] == config and bool(row["text"]) for row in rows)
        for config in CONFIGURATION_NAMES
    }
    requested_counts = {
        config: sum(row["resource_configuration"] == config for row in rows)
        for config in CONFIGURATION_NAMES
    }
    if requested_counts != {config: 120 for config in CONFIGURATION_NAMES}:
        raise AssertionError(f"unexpected task denominators: {requested_counts}")
    result = {
        "experiment": EXPERIMENT,
        "protocol_version": PROTOCOL_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "source_sha256": manifest["source_sha256"],
        "elapsed_wall_seconds": time.monotonic() - started,
        "workers": workers,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "host": platform.node(),
            "platform": f"{platform.system()} {platform.machine()}",
            "dependencies": manifest["runtime_dependencies"],
        },
        "task_count": len(rows),
        "paired_seed_band_blocks": len(pair_jobs),
        "basic_valid_output_count_by_configuration": valid_counts,
        "requested_task_count_by_configuration": requested_counts,
        "deadline_best_so_far_output_count": sum(row["status"] == "deadline_best_so_far_output" for row in rows),
        "deadline_no_valid_output_count": sum(row["status"] == "deadline_no_valid_output" for row in rows),
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output_path)
    return result


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def analyze(manifest_path: Path, result_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_manifest(manifest_path, manifest)
    result = json.loads(result_path.read_text())
    if result.get("protocol_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("result does not link to the frozen manifest")
    if result.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("result source hashes do not match the frozen manifest")
    if result.get("experiment") != EXPERIMENT or result.get("status") != "complete":
        raise ValueError("the supplied result is not a completed resource OAT run")
    if result.get("runtime", {}).get("python_version") != TARGET_PYTHON_VERSION:
        raise ValueError("result was not produced by the frozen Python runtime")
    if result.get("runtime", {}).get("python_implementation") != TARGET_PYTHON_IMPLEMENTATION:
        raise ValueError("result was not produced by the frozen Python implementation")
    if result.get("runtime", {}).get("platform") != f"{TARGET_SYSTEM} {TARGET_MACHINE}":
        raise ValueError("result was not produced on the frozen platform")
    if result.get("runtime", {}).get("dependencies") != manifest["runtime_dependencies"]:
        raise ValueError("result dependency versions differ from the frozen manifest")
    if not 1 <= result.get("workers", 0) <= 6:
        raise ValueError("result worker count is outside the frozen safe range")
    rows = result.get("results", [])
    expected_jobs = manifest["jobs"]
    if len(rows) != len(expected_jobs) or len(rows) != 600:
        raise ValueError("analysis requires every one of the 600 frozen task rows")
    task_fields = (
        "phase", "pair_id", "seed", "band_id", "min_letters", "max_letters",
        "method", "resource_configuration", "configuration_position",
        "execution_order_within_seed_band_block",
    )
    expected_by_key = {
        (job["seed"], job["band_id"], job["method"], job["resource_configuration"]): job
        for job in expected_jobs
    }
    actual_by_key = {}
    for row in rows:
        key = (row.get("seed"), row.get("band_id"), row.get("method"), row.get("resource_configuration"))
        if key in actual_by_key:
            raise ValueError(f"duplicate task row: {key}")
        actual_by_key[key] = row
    if actual_by_key.keys() != expected_by_key.keys():
        raise ValueError("raw result task keys do not match the frozen manifest")

    for key, job in expected_by_key.items():
        row = actual_by_key[key]
        if any(row.get(field) != job[field] for field in task_fields):
            raise ValueError(f"task row differs from its frozen job: {key}")
        if row.get("elapsed_seconds", -1) < 0:
            raise ValueError(f"negative or missing elapsed time: {key}")
        if row.get("eligible_exact_closure_candidates", -1) < 0:
            raise ValueError(f"negative or missing closure count: {key}")
        if row.get("basic_valid_closure_candidates", -1) < 0:
            raise ValueError(f"negative or missing valid-closure count: {key}")
        if row["basic_valid_closure_candidates"] > row["eligible_exact_closure_candidates"]:
            raise ValueError(f"valid closures exceed eligible closures: {key}")

        text = row.get("text")
        config = manifest["resource_configurations"][job["resource_configuration"]]
        timed_out = row["elapsed_seconds"] >= config["cooperative_budget_seconds"]
        if text:
            if row.get("text_sha256") != sha256_bytes(text.encode()):
                raise ValueError(f"selected text hash mismatch: {key}")
            normalized = BASE.normalize_letters(text)
            if not normalized or normalized != normalized[::-1]:
                raise ValueError(f"output failed independent letter-palindrome check: {key}")
            checks = BASE.mechanical_admission_checks(
                text, min_letters=job["min_letters"], max_letters=job["max_letters"]
            )
            basic = {name: checks[name] for name in BASE.STRUCTURAL_CHECKS}
            if row.get("basic_structural_checks") != basic or not all(basic.values()):
                raise ValueError(f"selected output fails or misreports the basic gate: {key}")
            if row.get("strict_admission_checks") != checks:
                raise ValueError(f"strict admission checks do not match recomputation: {key}")
            if row.get("strict_admitted_selected_output") is not all(checks.values()):
                raise ValueError(f"strict-admission summary mismatch: {key}")
            if row.get("letters") != len(normalized) or row.get("word_count") != len(text.split()):
                raise ValueError(f"selected output length metadata mismatch: {key}")
            if row.get("basic_valid_closure_candidates", 0) <= 0:
                raise ValueError(f"selected output has no recorded valid closure: {key}")
            expected_status = "deadline_best_so_far_output" if timed_out else "selected_basic_valid_output"
            if row.get("status") != expected_status:
                raise ValueError(f"selected-output status disagrees with its deadline: {key}")
            if row.get("independent_exactness_check") is not True:
                raise ValueError(f"independent exactness flag is false: {key}")
        else:
            if row.get("text_sha256") is not None:
                raise ValueError(f"missing-output row has a text hash: {key}")
            if row.get("strict_admission_checks") != {}:
                raise ValueError(f"missing-output row has strict checks: {key}")
            if row.get("strict_admitted_selected_output") is not None:
                raise ValueError(f"missing-output row has a strict-pass flag: {key}")
            if row.get("independent_exactness_check") is not None:
                raise ValueError(f"missing-output row has an exactness flag: {key}")
            if any(row.get("basic_structural_checks", {}).values()):
                raise ValueError(f"missing-output row has a passing basic check: {key}")
            expected_status = "deadline_no_valid_output" if timed_out else "no_basic_valid_output"
            if row.get("status") != expected_status:
                raise ValueError(f"missing-output status disagrees with its deadline: {key}")

    recomputed_valid = {
        config: sum(row["resource_configuration"] == config and bool(row["text"]) for row in rows)
        for config in CONFIGURATION_NAMES
    }
    recomputed_requested = {
        config: sum(row["resource_configuration"] == config for row in rows)
        for config in CONFIGURATION_NAMES
    }
    recomputed_best = sum(row["status"] == "deadline_best_so_far_output" for row in rows)
    recomputed_deadline_empty = sum(row["status"] == "deadline_no_valid_output" for row in rows)
    if result.get("task_count") != 600 or result.get("paired_seed_band_blocks") != 60:
        raise ValueError("result-level task or paired-block count mismatch")
    if result.get("basic_valid_output_count_by_configuration") != recomputed_valid:
        raise ValueError("result-level valid-output counts do not match raw rows")
    if result.get("requested_task_count_by_configuration") != recomputed_requested:
        raise ValueError("result-level denominators do not match raw rows")
    if result.get("deadline_best_so_far_output_count") != recomputed_best:
        raise ValueError("result-level best-so-far count does not match raw rows")
    if result.get("deadline_no_valid_output_count") != recomputed_deadline_empty:
        raise ValueError("result-level deadline-empty count does not match raw rows")

    counts: dict[str, dict] = {}
    for config in CONFIGURATION_NAMES:
        config_rows = [row for row in rows if row["resource_configuration"] == config]
        counts[config] = {
            "requested_tasks": len(config_rows),
            "basic_valid_outputs": sum(bool(row["text"]) for row in config_rows),
            "eligible_closure_tasks": sum(row["eligible_exact_closure_candidates"] > 0 for row in config_rows),
            "strict_admitted_selected_outputs": sum(row["strict_admitted_selected_output"] is True for row in config_rows),
            "deadline_best_so_far_outputs": sum(row["status"] == "deadline_best_so_far_output" for row in config_rows),
            "deadline_without_valid_output": sum(row["status"] == "deadline_no_valid_output" for row in config_rows),
            "median_elapsed_seconds": statistics.median(row["elapsed_seconds"] for row in config_rows),
            "by_method": {
                method: {
                    "requested_tasks": sum(row["method"] == method for row in config_rows),
                    "basic_valid_outputs": sum(row["method"] == method and bool(row["text"]) for row in config_rows),
                    "eligible_closure_tasks": sum(row["method"] == method and row["eligible_exact_closure_candidates"] > 0 for row in config_rows),
                }
                for method in METHODS
            },
        }

    seed_ids = sorted({row["seed"] for row in rows})
    variant_names = CONFIGURATION_NAMES[1:]
    per_seed_differences = {name: [] for name in variant_names}
    points = {}
    for config in variant_names:
        points[config] = (counts[config]["basic_valid_outputs"] - counts["original_v3"]["basic_valid_outputs"]) / 120.0
    for seed in seed_ids:
        seed_rows = [row for row in rows if row["seed"] == seed]
        baseline = sum(row["resource_configuration"] == "original_v3" and bool(row["text"]) for row in seed_rows)
        for config in variant_names:
            treatment = sum(row["resource_configuration"] == config and bool(row["text"]) for row in seed_rows)
            per_seed_differences[config].append((treatment - baseline) / 6.0)

    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = {name: [] for name in variant_names}
    max_absolute_deviations = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(seed_ids)) for _ in seed_ids]
        sampled = {
            name: sum(per_seed_differences[name][index] for index in indices) / len(indices)
            for name in variant_names
        }
        for name in variant_names:
            bootstrap[name].append(sampled[name])
        max_absolute_deviations.append(max(abs(sampled[name] - points[name]) for name in variant_names))
    simultaneous_radius = _percentile(max_absolute_deviations, 0.95)
    contrasts = {
        name: {
            "treatment": name,
            "control": "original_v3",
            "paired_difference_in_basic_valid_coverage": points[name],
            "seed_cluster_bootstrap_95_percentile_interval_unadjusted": [
                _percentile(bootstrap[name], 0.025), _percentile(bootstrap[name], 0.975)
            ],
            "familywise_95_percentile_interval_max_absolute_deviation": [
                max(-1.0, points[name] - simultaneous_radius),
                min(1.0, points[name] + simultaneous_radius),
            ],
        }
        for name in variant_names
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "post_hoc_exploratory_resource_one_factor_screen",
        "interpretation_limit": manifest["design"]["claim_boundary"],
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "raw_results_sha256": sha256_file(result_path),
        "source_sha256": manifest["source_sha256"],
        "task_count": len(rows),
        "counts": counts,
        "contrasts": contrasts,
        "bootstrap": {
            "seed_clusters": len(seed_ids),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "simultaneous_method": "95th percentile of the maximum absolute deviation across the four paired contrasts in each shared seed-cluster bootstrap replicate",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    report_path = output_path.with_suffix(".md")
    config_labels = {
        "original_v3": "Original v3 settings",
        "beam_width_256_only": "Beam width 256 only",
        "candidate_limit_800_only": "Candidate limit 800 only",
        "per_parent_32_only": "Per-parent branching 32 only",
        "budget_180_seconds_only": "180-second budget only",
    }
    result_rows = "\n".join(
        f"| {config_labels[name]} | {counts[name]['basic_valid_outputs']}/120 | "
        f"{counts[name]['eligible_closure_tasks']} | {counts[name]['strict_admitted_selected_outputs']} | "
        f"{counts[name]['median_elapsed_seconds']:.3f} |"
        for name in CONFIGURATION_NAMES
    )
    contrast_rows = "\n".join(
        f"| {config_labels[name]} | {contrasts[name]['paired_difference_in_basic_valid_coverage']:.3f} | "
        f"[{contrasts[name]['seed_cluster_bootstrap_95_percentile_interval_unadjusted'][0]:.3f}, "
        f"{contrasts[name]['seed_cluster_bootstrap_95_percentile_interval_unadjusted'][1]:.3f}] | "
        f"[{contrasts[name]['familywise_95_percentile_interval_max_absolute_deviation'][0]:.3f}, "
        f"{contrasts[name]['familywise_95_percentile_interval_max_absolute_deviation'][1]:.3f}] |"
        for name in variant_names
    )
    report = f"""# Resource one-factor-at-a-time search screen (exploratory)

**Status:** post-hoc exploratory diagnosis. The four factors were selected after observing the v4 resource-bundle result. All contrasts are reported; the estimates do not establish a confirmatory result or readability.

## Frozen comparison

Twenty new seeds (5000–5019), three normalized-letter length bands, and both native algorithms give 120 requested tasks per configuration. The five conditions are the original v3 settings plus four one-setting variants: beam width 256, candidate limit 800, per-parent branching 32, or a 180-second cooperative budget. Each variant holds the other settings at their original-v3 values. The same vocabulary, scorer, methods, gates, and task identifiers are used throughout.

## Results

| Configuration | Basic-valid outputs / requested | Eligible-closure tasks | Strict-pass selected outputs | Median seconds/task |
|---|---:|---:|---:|---:|
{result_rows}

## Paired contrasts

Positive values favor the one-setting variant. Intervals resample the 20 seed clusters, not 600 individual tasks. Unadjusted percentile intervals are shown alongside a familywise 95% max-absolute-deviation interval for the four contrasts. Treat these as exploratory descriptions: the factors were selected after v4, and this screen cannot detect interactions among the four settings.

| Variant vs original v3 | Coverage difference | Unadjusted seed-cluster 95% interval | Familywise 95% interval |
|---|---:|---:|---:|
{contrast_rows}

## Claim boundary

This estimates each setting's marginal effect at the original-v3 values on this implementation, vocabulary, and task set. It does not identify interactions or explain the combined v4 result on its own. It does not measure grammar, understandability, or human preference.

## Provenance

- Frozen manifest SHA-256: `{summary['protocol_manifest_sha256']}`
- Raw results SHA-256: `{summary['raw_results_sha256']}`
- Analysis JSON SHA-256: `{sha256_file(output_path)}`
- Runner: `{manifest['sources']['ablation_protocol']}`
"""
    report_path.write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST_DEFAULT)
    prepare_parser.add_argument("--output", type=Path, default=RESULT_DIRECTORY / "manifest.json")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=6)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser.add_argument("--results", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.source_manifest, args.output)
        print(json.dumps({
            "manifest": str(args.output),
            "manifest_sha256": sha256_file(args.output),
            "jobs": len(manifest["jobs"]),
            "per_configuration": manifest["task_count_by_configuration"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
        }, indent=2))
    elif args.command == "run":
        result = run(args.manifest, args.output, args.workers)
        print(json.dumps({
            "status": result["status"],
            "tasks": result["task_count"],
            "outputs": result["basic_valid_output_count_by_configuration"],
            "wall_seconds": result["elapsed_wall_seconds"],
            "output": str(args.output),
        }, indent=2))
    else:
        summary = analyze(args.manifest, args.results, args.output)
        print(json.dumps({
            "contrasts": summary["contrasts"],
            "counts": summary["counts"],
            "report": str(args.output.with_suffix(".md")),
        }, indent=2))


if __name__ == "__main__":
    main()
