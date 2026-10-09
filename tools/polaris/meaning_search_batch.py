"""Bounded, checkpointed CPU search and GPU proposal lanes on one PBS node."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def job_specs(out, gpu_python, profile="initial", model_cache=None):
    if profile not in ("initial", "expanded"):
        raise ValueError("unknown batch profile")
    jobs = []
    for seed in (20261009, 20261010, 20261011):
        for lane, script, seconds, ceiling in (
            ("connected-events", "argonne_connected_events_20261009.py", 120, 150),
            ("typed-overhang", "argonne_typed_overhang_growth_20261009.py", 120, 525),
        ):
            expanded = profile == "expanded"
            if expanded:
                seconds, ceiling = 120, (150 if lane == "connected-events" else 525)
            name = f"{lane}-{seed}"
            command = ["/usr/bin/python3.11", str(ROOT / "experiments" / script),
                       "--out", str(out / f"{name}.json"), "--seconds", str(seconds),
                       "--seed", str(seed)]
            if lane == "typed-overhang":
                command += ["--bigram-file", str(ROOT / "data/count_2w.txt")]
                if expanded:
                    command += ["--max-letters", "2500", "--beam-width", "384", "--max-steps", "1500"]
            elif expanded:
                command += ["--max-letters", "2500", "--max-states", "1000000"]
            jobs.append({"id": name, "command": command, "hard_timeout_seconds": ceiling})
        if profile == "expanded":
            name = f"free-typed-{seed}"
            jobs.append({"id": name, "hard_timeout_seconds": 525,
                         "command": ["/usr/bin/python3.11", str(ROOT / "experiments/argonne_free_typed_clauses_20261009.py"),
                                     "--out", str(out / f"{name}.json"), "--seconds", "120", "--seed", str(seed),
                                     "--bigram-file", str(ROOT / "data/count_2w.txt"), "--beam-width", "384",
                                     "--max-letters", "2500", "--max-steps", "1500", "--max-overhang", "96"]})
    model_seconds = "450" if profile == "expanded" else "900"
    jobs.append({"id": "model-meaning-proposals", "hard_timeout_seconds": 500 if profile == "expanded" else 1200,
                 "command": [gpu_python, str(ROOT / "experiments/argonne_model_meaning_proposals_20261009.py"),
                             "--out", str(out / "model-meaning-proposals.json"),
                             "--model", "Qwen/Qwen3-8B", "--revision", REVISION,
                             "--attempts", "24", "--max-new-tokens", "768", "--seconds", model_seconds]})
    if model_cache:
        jobs[-1]["command"] += ["--cache-dir", model_cache]
    return jobs


def run_job(spec, out):
    started = time.monotonic()
    with (out / f"{spec['id']}.log").open("w") as log:
        try:
            process = subprocess.run(spec["command"], cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT,
                                     timeout=spec["hard_timeout_seconds"], check=False)
            row = {**spec, "returncode": process.returncode,
                   "status": "completed" if process.returncode == 0 else "execution_error"}
        except subprocess.TimeoutExpired:
            row = {**spec, "returncode": None, "status": "hard_timeout"}
        except Exception as exc:
            row = {**spec, "returncode": None, "status": "launch_error", "error": f"{type(exc).__name__}: {exc}"}
    row["elapsed_seconds"] = time.monotonic() - started
    row["result_exists"] = (out / f"{spec['id']}.json").exists()
    return row


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu-python", required=True)
    parser.add_argument("--profile", choices=("initial", "expanded"), default="initial")
    parser.add_argument("--model-cache")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    specs = job_specs(args.out.resolve(), args.gpu_python, args.profile, args.model_cache)
    manifest = ROOT / "source-manifest.json"
    payload = {"experiment": "argonne-length-meaning-20261009", "status": "running",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "pbs_job_id": os.environ.get("PBS_JOBID"),
               "profile": args.profile,
               "compute_host": os.uname().nodename, "driver_python": sys.version,
               "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "planned": specs, "results": [],
               "privacy": "Public research code, credited Reddit titles, public lexical counts and published exemplars only; no private reader ratings.",
               "acceptance": "Whole-text exactness, then independent whole-passage meaning and grammar evaluation; model self-ratings and symbolic role checks are not reader acceptance."}
    summary = args.out / "batch.json"
    save(summary, payload)
    with ThreadPoolExecutor(max_workers=len(specs)) as executor:
        futures = [executor.submit(run_job, spec, args.out) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            payload["results"].append(row)
            payload["elapsed_seconds"] = time.monotonic() - started
            save(summary, payload)
            print(json.dumps({key: row[key] for key in ("id", "status", "elapsed_seconds", "result_exists")}), flush=True)
    payload.update(status="completed_bounded_batch", elapsed_seconds=time.monotonic() - started)
    save(summary, payload)


if __name__ == "__main__":
    main()
