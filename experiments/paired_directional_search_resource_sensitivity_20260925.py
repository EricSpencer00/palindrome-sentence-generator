"""Frozen v4 resource-sensitivity follow-up for palindrome search.

The outside-in arm follows the Norvig/Hoey residual-search principle. The
center-out arm is the project comparator. This is an operational comparison
of two configured algorithms under their native closure rules; it is not a
causal isolation of search direction or a reproduction of either cited system.
All language diagnostics are reported as automatic model evidence, never as
human readability ratings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.centerout import centerout_search  # noqa: E402
from llm_palindrome.generate import ZipfScorer  # noqa: E402
from llm_palindrome.lexicon import is_real_word, load_lexicon  # noqa: E402
from llm_palindrome.search import State, WordTries, beam_search  # noqa: E402
from llm_palindrome.shortwords import is_real_short  # noqa: E402


EXPERIMENT = "paired-directional-search-resource-sensitivity-20260925"
VOCAB_SOURCE = ROOT / "tools/polaris/payload/vocab30k.txt"
LEXICON_SOURCE = ROOT / "data/lexicon.txt"
HISTORY_DIR = ROOT / "runs/directional-search-benchmark-20260925"
LENGTH_BANDS = ((30, 49), (50, 79), (80, 119))
MAX_WORDS = 32
RESOURCE_CONFIGURATIONS = {
    "original_v3": {
        "beam_width": 60,
        "candidate_limit": 200,
        "per_parent": 8,
        "max_steps": 400,
        "diversity": 0.4,
        "max_overhang": 24,
        "max_words": MAX_WORDS,
        "cooperative_budget_seconds": 45,
    },
    "escalated_v4": {
        "beam_width": 256,
        "candidate_limit": 800,
        "per_parent": 32,
        "max_steps": 400,
        "diversity": 0.4,
        "max_overhang": 24,
        "max_words": MAX_WORDS,
        "cooperative_budget_seconds": 180,
    },
}
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_REPLICATES = 20000
METHODS = ("outside_in_norvig_hoey_adaptation", "center_out_project")
STRUCTURAL_CHECKS = (
    "supported_ascii_letters",
    "nonempty",
    "exact_letter_palindrome",
    "length_band",
    "word_form",
    "lexicon_words",
    "ordinary_short_words",
    "not_forbidden_catalogue_control",
    "not_catalogue_family_derivative",
    "not_forbidden_catalogue_endpoint_scaffold",
    "absent_from_local_catalogue",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def prepare_manifest(output_dir: Path) -> dict:
    """Freeze the ranked vocabulary and all task/settings before inference."""
    if not VOCAB_SOURCE.is_file() or not LEXICON_SOURCE.is_file():
        raise FileNotFoundError("the pinned vocabulary or project lexicon is missing")
    lexicon = load_lexicon(str(LEXICON_SOURCE))
    source_words = VOCAB_SOURCE.read_text().splitlines()
    vocabulary: list[str] = []
    seen: set[str] = set()
    for raw in source_words:
        word = raw.strip().lower()
        if (not word or not word.isascii() or not word.isalpha()
                or word in seen or not is_real_word(word, lexicon)
                or not is_real_short(word)):
            continue
        vocabulary.append(word)
        seen.add(word)
    if len(vocabulary) < 1000:
        raise ValueError(f"unexpectedly small frozen vocabulary: {len(vocabulary)}")

    seeds = {
        "development": [3000, 3001],
        "confirmatory": list(range(4000, 4020)),
    }
    jobs = []
    configuration_names = tuple(RESOURCE_CONFIGURATIONS)
    for phase, phase_seeds in seeds.items():
        for seed in phase_seeds:
            for band_index, (minimum, maximum) in enumerate(LENGTH_BANDS, 1):
                pair_id = f"{phase}:seed-{seed}:band-{band_index}"
                configuration_order = (
                    configuration_names if (seed + band_index) % 2 == 0
                    else tuple(reversed(configuration_names))
                )
                execution_order = 0
                for config_position, config_name in enumerate(configuration_order):
                    method_order = (
                        METHODS if (seed + band_index + config_position) % 2 == 0
                        else tuple(reversed(METHODS))
                    )
                    for method in method_order:
                        execution_order += 1
                        jobs.append({
                            "phase": phase,
                            "pair_id": pair_id,
                            "seed": seed,
                            "band_id": band_index,
                            "min_letters": minimum,
                            "max_letters": maximum,
                            "method": method,
                            "resource_configuration": config_name,
                            "configuration_position": config_position + 1,
                            "execution_order_within_seed_band_block": execution_order,
                        })

    source_paths = {
        "experiment_script": "experiments/paired_directional_search_resource_sensitivity_20260925.py",
        "brown_diagnostic": "experiments/score_length_stratified_readability.py",
        "brown_v4_scorer": "experiments/score_directional_search_brown_20260925.py",
        "brown_model_source": "experiments/audit_programmatic_readability.py",
        "gpt2_verifier": "experiments/score_gpt2_static_verifier_20260925.py",
        "gpt2_requirements": "experiments/static-verifier-nuc-requirements.txt",
        "deepseek_verifier": "experiments/infer_deepseek_static_verifier_20260925.py",
        "deepseek_requirements": "experiments/static-verifier-bedrock-requirements.txt",
        "static_verifier_helper": "experiments/static_language_verifier_20260925.py",
        "verifier_input_builder": "experiments/prepare_directional_static_verifier_20260925.py",
        "readability_requirements": "paper/readability-requirements.txt",
        "toy_correctness_controls": "tests/test_initial_state_search.py",
        "vocabulary_file": VOCAB_SOURCE.relative_to(ROOT).as_posix(),
        "project_lexicon": LEXICON_SOURCE.relative_to(ROOT).as_posix(),
        "known_palindromes": "data/known_palindromes.json",
        "search_implementation": "llm_palindrome/search.py",
        "centerout_implementation": "llm_palindrome/centerout.py",
        "score_implementation": "llm_palindrome/generate.py",
        "admission_implementation": "llm_palindrome/admission.py",
        "lexicon_implementation": "llm_palindrome/lexicon.py",
        "shortword_implementation": "llm_palindrome/shortwords.py",
        "scoring_implementation": "llm_palindrome/scoring.py",
    }
    source_hashes = {
        name: sha256_file(ROOT / path) for name, path in source_paths.items()
    }
    citations = {
        "outside_in_algorithm_reference": "https://www.norvig.com/pal-alg.html",
        "brown_corpus": "https://www.nltk.org/nltk_data/",
        "gpt2_paper": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "gpt2_model": "https://huggingface.co/openai-community/gpt2",
        "deepseek_bedrock_model_card": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html",
        "deepseek_bedrock_pricing": "https://aws.amazon.com/bedrock/pricing/",
        "prior_computational_palindrome_work": "https://www.ijcai.org/Proceedings/15/Papers/353.pdf",
    }
    previous_run_specs = (
        (1, HISTORY_DIR / "manifest-v1-pilot.json", HISTORY_DIR / "development-v1-pilot.json"),
        (2, HISTORY_DIR / "manifest-v2-empty-debt-pilot.json", HISTORY_DIR / "development-v2-empty-debt-pilot.json"),
        (3, HISTORY_DIR / "manifest.json", HISTORY_DIR / "development-v3-native-closure.json"),
        (3, HISTORY_DIR / "manifest.json", HISTORY_DIR / "confirmatory-v3-native-closure.json"),
    )
    previous_runs = []
    for prior_version, manifest_path, result_path in previous_run_specs:
        if not manifest_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(f"prior protocol v{prior_version} result archive is missing")
        previous_result = json.loads(result_path.read_text())
        previous_runs.append({
            "protocol_version": prior_version,
            "manifest_path_from_repository_root": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "result_path_from_repository_root": result_path.relative_to(ROOT).as_posix(),
            "result_sha256": sha256_file(result_path),
            "task_count": len(previous_result["results"]),
            "basic_valid_output_count": sum(bool(row.get("text")) for row in previous_result["results"]),
            "role_in_v4": "retained as prior evidence; not included in v4 estimates",
        })
    result = {
        "experiment": EXPERIMENT,
        "protocol_version": 4,
        "status": "adaptive_resource_followup_frozen_before_v4_development_and_confirmatory_runs",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "unit": "resource-configuration comparison on paired operational algorithm tasks; both algorithms run under each configuration on every shared task",
            "independent_cluster": "search seed; all three bands, algorithms, and resource configurations for one seed remain together in bootstrap resampling",
            "development_seeds": seeds["development"],
            "confirmatory_seeds": seeds["confirmatory"],
            "length_bands_normalized_ascii_letters_inclusive": [list(x) for x in LENGTH_BANDS],
            "fresh_seed_block": "all v4 seeds were chosen after v3 and are disjoint from v3 seeds",
            "motivation_and_adaptive_status": "v3 returned zero eligible closures in both arms; v4 is an adaptive resource-sensitivity follow-up, not an independent confirmation of a hypothesis fixed before v3",
            "prior_v3_result": previous_runs[-1],
            "correctness_controls": "toy-vocabulary tests confirm that both native search APIs reach the known exact target 'level level', that the target passes the same state and basic structural gates, and that each native closure callback fires; candidate_limit=4 and per_parent=4 expose the required word",
            "initialization": "both searches start from the same empty text; the center-out arm is explicitly the empty-center configuration",
            "arm_order": "all four method-by-resource conditions for each seed-by-length block run sequentially in one worker; resource-configuration order and algorithm order alternate across blocks using frozen parity rules",
            "native_closure_rules": {
                "outside_in_norvig_hoey_adaptation": "closes when the residual character debt is itself a palindrome; this is the standard residual-search closure condition",
                "center_out_project": "closes only when its outer-edge debt is exactly empty",
            },
            "comparison_scope": "resource-configuration comparison, not a runtime-only or causal effect: beam width, candidate limit, per-parent branching, and cooperative budget all change together; algorithm comparisons remain secondary and descriptive",
            "text_rendering": "lowercase space-joined words, initial character capitalized, one final period",
            "state_constraints": "same per-method state gate: at most 32 whitespace-delimited words, repeated words are permitted and retained for shortcut analysis, only frozen-vocabulary words, and no partial state longer than the band's upper bound",
            "basic_structural_gate": list(STRUCTURAL_CHECKS),
            "strict_gate": "all checks returned true by mechanical_admission_checks; reported only as a selected-output sensitivity analysis",
            "primary_endpoint": "paired difference in basic-valid output coverage between escalated_v4 and original_v3 resource configurations, pooled across both methods; denominator is 120 requested tasks per configuration (20 seeds x 3 bands x 2 methods)",
            "secondary_algorithm_endpoint": "within each resource configuration, describe outside-in versus empty-center coverage; do not attribute differences to direction alone because native closure spaces differ",
            "closure_endpoint": "fraction of requested tasks that reach at least one eligible native exact-palindrome closure at or above the band's minimum length, before the structural gate; closure and basic-valid counts are reported separately",
            "strict_gate_endpoint": "fraction of selected basic-valid outputs passing all strict checks; this is not a search-level strict-coverage rate",
            "secondary_language_endpoints": [
                "Brown add-0.1 word-bigram order gain over 32 deterministic same-word shuffles; the scorer is trained on 80% of Brown documents and candidates are not used for training",
                "pinned GPT-2 mean causal log probability per predicted token for each selected output and its 32 same-word shuffles; fixed model revision 607a30d783dfa663caf39e06633721c8d4cfcd7e",
                "one counterbalanced DeepSeek V3.2 choice per selected output against deterministic same-word shuffle replicate 00; AWS Bedrock us-east-2, temperature 0, top-p 1, max 256 output tokens, total spend preflight capped at $0.20",
            ],
            "selection_rule": "each search returns the eligible closure maximizing accumulated ZipfScorer score divided by normalized-letter count; Brown, GPT-2, DeepSeek, and strict-gate outcomes do not select or replace outputs",
            "shortcut_handling": "retain basic-valid outputs even when strict checks identify repetition, word-order symmetry, or other shortcuts; report overall and strict-pass/fail strata",
            "failures": "all task/method failures and timeouts remain in the result denominator",
            "uncertainty": f"paired percentile bootstrap resampling {BOOTSTRAP_REPLICATES} whole seed clusters with replacement; PRNG seed {BOOTSTRAP_SEED}; with no successes, the interval is non-informative and must not be interpreted as equivalence",
            "resource_limit": "six NUC workers; cooperative per-job budgets and all other resource settings are frozen per configuration; an in-flight operation can slightly overrun a checkpoint; Bedrock spend guard is $0.20 and inference runs only for selected valid outputs",
            "language_interpretation": "Brown and GPT-2 are automatic diagnostics of corpus word-order fit or model likelihood; DeepSeek is one prompted model's preference, not human evidence; language-score paired differences are computed only for jointly successful matched tasks",
            "previous_protocol_results": previous_runs,
        },
        "resource_configurations": RESOURCE_CONFIGURATIONS,
        "max_words": MAX_WORDS,
        "vocabulary": {
            "source_path": source_paths["vocabulary_file"],
            "source_sha256": source_hashes["vocabulary_file"],
            "source_line_count": len(source_words),
            "filtered_word_count": len(vocabulary),
            "filtered_word_sha256": sha256_bytes(("\n".join(vocabulary) + "\n").encode()),
            "filter": "ASCII alphabetic, stable first occurrence, project lexicon headword or regular inflection, and project ordinary-short-word predicate",
            "words": vocabulary,
        },
        "sources": source_paths,
        "citations": citations,
        "source_sha256": source_hashes,
        "jobs": jobs,
        "provenance": {
            "manifest_preparation_python_version": platform.python_version(),
            "manifest_preparation_host": platform.node(),
            "project_git_commit_at_freeze": __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "wordfreq_required_version": "3.1.1 (pinned by paper/readability-requirements.txt)",
            "model_judge": "AWS Bedrock DeepSeek V3.2, temperature 0, top-p 1; provider seed unavailable, so raw requests and replies will be retained",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def render(words: tuple[str, ...] | list[str]) -> str:
    text = " ".join(words).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def state_gate(words: tuple[str, ...], maximum: int) -> bool:
    if not words:
        return True
    flattened = [part.lower() for unit in words for part in unit.split()]
    if len(flattened) > MAX_WORDS:
        return False
    letter_count = sum(len(normalize_letters(word)) for word in flattened)
    if letter_count > maximum:
        return False
    return True


def run_job(job: dict[str, Any], tries: WordTries, scorer: ZipfScorer) -> dict:
    minimum, maximum = job["min_letters"], job["max_letters"]
    settings = RESOURCE_CONFIGURATIONS[job["resource_configuration"]]
    counters: Counter[str] = Counter()
    failed_checks: Counter[str] = Counter()
    started = time.monotonic()
    deadline = started + settings["cooperative_budget_seconds"]

    def allow_state(left, right):
        return state_gate(tuple(left) + tuple(right), maximum)

    def allow_closed(left, right):
        if time.monotonic() >= deadline:
            counters["closure_candidates_at_deadline"] += 1
            return False
        counters["eligible_exact_closure_candidates"] += 1
        words = tuple(left) + tuple(right)
        text = render(words)
        checks = mechanical_admission_checks(
            text, min_letters=minimum, max_letters=maximum,
        )
        for name, passed in checks.items():
            if name in STRUCTURAL_CHECKS and not passed:
                failed_checks[name] += 1
        basic_valid = all(checks[name] for name in STRUCTURAL_CHECKS)
        if basic_valid:
            counters["basic_valid_closure_candidates"] += 1
        return basic_valid

    if job["method"] == METHODS[0]:
        initial = State(0.0, (), (), "", "L", 0.0)
        words = beam_search(
            tries, scorer,
            min_letters=minimum,
            beam_width=settings["beam_width"],
            max_steps=settings["max_steps"],
            candidate_limit=settings["candidate_limit"],
            per_parent=settings["per_parent"],
            seed=job["seed"],
            diversity=settings["diversity"],
            max_word_uses=None,
            initial_state=initial,
            allow_state=allow_state,
            allow_closed=allow_closed,
            deadline=deadline,
        )
    elif job["method"] == METHODS[1]:
        words = centerout_search(
            tries, scorer,
            min_letters=minimum,
            beam_width=settings["beam_width"],
            center="",
            max_steps=settings["max_steps"],
            candidate_limit=settings["candidate_limit"],
            per_parent=settings["per_parent"],
            seed=job["seed"],
            diversity=settings["diversity"],
            max_overhang=settings["max_overhang"],
            allow_state=allow_state,
            allow_closed=allow_closed,
            deadline=deadline,
        )
    else:
        raise ValueError(f"unknown method: {job['method']}")
    elapsed = time.monotonic() - started
    timed_out = time.monotonic() >= deadline
    text = render(words)
    checks = mechanical_admission_checks(
        text, min_letters=minimum, max_letters=maximum,
    ) if text else {}
    basic_checks = {name: checks.get(name, False) for name in STRUCTURAL_CHECKS}
    basic_valid = bool(text and all(basic_checks.values()))
    if text and not basic_valid:
        raise AssertionError("search returned an output rejected by its shared basic closure gate")
    strict_admitted = bool(text and all(checks.values()))
    tape = normalize_letters(text)
    return {
        **job,
        "status": (
            "deadline_best_so_far_output" if timed_out and basic_valid else
            "deadline_no_valid_output" if timed_out else
            "selected_basic_valid_output" if basic_valid else "no_basic_valid_output"
        ),
        "elapsed_seconds": elapsed,
        "letters": len(tape),
        "word_count": len(words),
        "text": text if basic_valid else None,
        "text_sha256": sha256_bytes(text.encode()) if basic_valid else None,
        "independent_exactness_check": bool(tape and tape == tape[::-1]) if basic_valid else None,
        "basic_structural_checks": basic_checks,
        "strict_admission_checks": checks if basic_valid else {},
        "strict_admitted_selected_output": strict_admitted if basic_valid else None,
        "eligible_exact_closure_candidates": counters["eligible_exact_closure_candidates"],
        "basic_valid_closure_candidates": counters["basic_valid_closure_candidates"],
        "closure_candidates_at_deadline": counters["closure_candidates_at_deadline"],
        "rejected_closure_checks": dict(sorted(failed_checks.items())),
    }


_WORKER_TRIES: WordTries | None = None
_WORKER_SCORER: ZipfScorer | None = None


def _worker_init(manifest_path: str) -> None:
    global _WORKER_TRIES, _WORKER_SCORER
    manifest = json.loads(Path(manifest_path).read_text())
    _WORKER_TRIES = WordTries(manifest["vocabulary"]["words"])
    _WORKER_SCORER = ZipfScorer()


def _worker_run(job: dict) -> dict:
    if _WORKER_TRIES is None or _WORKER_SCORER is None:
        raise RuntimeError("worker was not initialized")
    return run_job(job, _WORKER_TRIES, _WORKER_SCORER)


def _worker_run_pair(pair_jobs: list[dict]) -> list[dict]:
    if _WORKER_TRIES is None or _WORKER_SCORER is None:
        raise RuntimeError("worker was not initialized")
    return [run_job(job, _WORKER_TRIES, _WORKER_SCORER) for job in pair_jobs]


def verify_frozen_manifest(manifest: dict) -> dict:
    """Fail closed if code, vocabulary, settings, or worker dependency drifted."""
    if manifest.get("protocol_version") != 4:
        raise ValueError("execution requires the frozen v4 resource-sensitivity protocol")
    if manifest.get("resource_configurations") != RESOURCE_CONFIGURATIONS:
        raise ValueError("current resource configurations do not match the frozen manifest")
    if manifest.get("max_words") != MAX_WORDS:
        raise ValueError("current word-count cap does not match the frozen manifest")
    for name, relative_path in manifest["sources"].items():
        expected = manifest["source_sha256"].get(name)
        actual = sha256_file(ROOT / relative_path)
        if not expected or actual != expected:
            raise ValueError(f"frozen source hash mismatch: {name}")
    vocabulary = manifest["vocabulary"]["words"]
    observed_vocabulary_hash = sha256_bytes(("\n".join(vocabulary) + "\n").encode())
    if observed_vocabulary_hash != manifest["vocabulary"]["filtered_word_sha256"]:
        raise ValueError("frozen vocabulary hash mismatch")
    wordfreq_version = importlib.metadata.version("wordfreq")
    expected_wordfreq = manifest["provenance"]["wordfreq_required_version"].split()[0]
    if wordfreq_version != expected_wordfreq:
        raise ValueError(
            f"wordfreq version mismatch: expected {expected_wordfreq}, found {wordfreq_version}"
        )
    return {"wordfreq_version": wordfreq_version}


def run_phase(manifest_path: Path, output_path: Path, phase: str,
              workers: int = 6) -> dict:
    manifest = json.loads(manifest_path.read_text())
    runtime = verify_frozen_manifest(manifest)
    protocol_jobs = [row for row in manifest["jobs"] if row["phase"] == phase]
    if not protocol_jobs:
        raise ValueError(f"manifest has no jobs for phase {phase!r}")
    context = multiprocessing.get_context("fork")
    pairs: dict[str, list[dict]] = {}
    for job in protocol_jobs:
        pairs.setdefault(job["pair_id"], []).append(job)
    pair_jobs = list(pairs.values())
    expected_conditions = {
        (resource_configuration, method)
        for resource_configuration in RESOURCE_CONFIGURATIONS
        for method in METHODS
    }
    for pair in pair_jobs:
        observed_conditions = {
            (job["resource_configuration"], job["method"]) for job in pair
        }
        if (len(pair) != len(expected_conditions)
                or observed_conditions != expected_conditions
                or [job["execution_order_within_seed_band_block"] for job in pair]
                != list(range(1, len(expected_conditions) + 1))):
            raise ValueError(f"malformed v4 conditions in block: {pair[0]['pair_id']}")
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(str(manifest_path),),
    ) as pool:
        paired_rows = list(pool.map(_worker_run_pair, pair_jobs, chunksize=1))
    rows = list(itertools.chain.from_iterable(paired_rows))
    output = {
        "experiment": EXPERIMENT,
        "phase": phase,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_manifest_sha256": sha256_file(manifest_path),
        "elapsed_wall_seconds": time.monotonic() - started,
        "workers": workers,
        "runtime": {
            **runtime,
            "python_version": platform.python_version(),
            "host": platform.node(),
        },
        "task_count": len(rows),
        "seed_band_block_count": len(pair_jobs),
        "basic_valid_output_count": sum(bool(row["text"]) for row in rows),
        "deadline_best_so_far_output_count": sum(
            row["status"] == "deadline_best_so_far_output" for row in rows
        ),
        "deadline_no_valid_output_count": sum(
            row["status"] == "deadline_no_valid_output" for row in rows
        ),
        "basic_valid_output_count_by_resource_configuration": {
            configuration: sum(
                row["resource_configuration"] == configuration and bool(row["text"])
                for row in rows
            )
            for configuration in RESOURCE_CONFIGURATIONS
        },
        "requested_task_count_by_resource_configuration": {
            configuration: sum(row["resource_configuration"] == configuration for row in rows)
            for configuration in RESOURCE_CONFIGURATIONS
        },
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "runs/directional-search-resource-sensitivity-20260925",
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--phase", choices=("development", "confirmatory"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.command == "prepare":
        manifest = prepare_manifest(args.output_dir)
        print(json.dumps({
            "manifest": str(args.output_dir / "manifest.json"),
            "sha256": sha256_file(args.output_dir / "manifest.json"),
            "vocabulary_words": manifest["vocabulary"]["filtered_word_count"],
            "development_jobs": sum(x["phase"] == "development" for x in manifest["jobs"]),
            "confirmatory_jobs": sum(x["phase"] == "confirmatory" for x in manifest["jobs"]),
        }, indent=2))
        return
    result = run_phase(args.manifest, args.output, args.phase, args.workers)
    print(json.dumps({
        "phase": args.phase,
        "status": result["status"],
        "tasks": len(result["results"]),
        "basic_valid_outputs": result["basic_valid_output_count"],
        "basic_valid_outputs_by_resource_configuration": result[
            "basic_valid_output_count_by_resource_configuration"
        ],
        "deadline_best_so_far_outputs": result["deadline_best_so_far_output_count"],
        "deadline_without_output": result["deadline_no_valid_output_count"],
        "wall_seconds": result["elapsed_wall_seconds"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
