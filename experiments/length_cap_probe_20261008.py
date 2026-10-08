"""Frozen word-cap feasibility probe; exact construction is not readability.

Compare caps of 32 and 256 words with identical vocabulary, scorers, seeds,
search limits, and admission checks. Run substantive search on the NUC.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/length-cap-probe-20261008"
SEEDS = (9000, 9001)
BANDS = ((120, 159), (800, 999))
CAPS = (32, 256)
METHODS = ("outside_in", "center_out")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze():
    prior_path = ROOT / "runs/fragment-penalty-20261002/manifest.json"
    prior = json.loads(prior_path.read_text())
    jobs = []
    for seed, band, method in itertools.product(SEEDS, BANDS, METHODS):
        caps = list(CAPS)
        random.Random(f"20261008:{seed}:{band}:{method}").shuffle(caps)
        for order, cap in enumerate(caps):
            jobs.append(dict(id=f"s{seed}-n{band[0]}-{method}-w{cap}",
                             seed=seed, minimum=band[0], maximum=band[1],
                             method=method, max_words=cap, order=order))
    sources = [p.relative_to(ROOT).as_posix() for p in (ROOT / "llm_palindrome").glob("*.py")]
    sources += ["experiments/length_cap_probe_20261008.py", "data/lexicon.txt",
                "data/known_palindromes.json"]
    manifest = dict(experiment="length-cap-probe-20261008", public_paper_version="v1",
                    frozen_utc=datetime.now(timezone.utc).isoformat(), seeds=SEEDS,
                    bands=BANDS, caps=CAPS, methods=METHODS, jobs=jobs,
                    vocabulary=prior["vocabulary"], prior_manifest_sha256=digest(prior_path),
                    settings=dict(beam_width=256, candidate_limit=800, per_parent=32,
                                  max_steps=400, diversity=0.4, max_overhang=24,
                                  cooperative_budget_seconds=60),
                    source_sha256={p:digest(ROOT/p) for p in sorted(sources)},
                    objective="return independently exact outputs in each frozen band; compare only the word cap",
                    selection="baseline Zipf score per letter; no fragment intervention or language-model selection",
                    interpretation="exploratory cap-feasibility probe, two seeds; no external generator or readability advantage; all failures retained")
    path = OUT / "manifest.json"
    if path.exists():
        raise FileExistsError("frozen protocol already exists")
    save(path, manifest)
    print(json.dumps({"jobs":len(jobs), "manifest_sha256":digest(path)}))


def run_job(job, manifest):
    from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
    from llm_palindrome.centerout import centerout_search
    from llm_palindrome.generate import ZipfScorer
    from llm_palindrome.search import State, WordTries, beam_search
    basic_names = ("supported_ascii_letters", "nonempty", "exact_letter_palindrome",
                   "length_band", "word_form", "lexicon_words", "ordinary_short_words",
                   "not_forbidden_catalogue_control", "not_catalogue_family_derivative",
                   "not_forbidden_catalogue_endpoint_scaffold", "absent_from_local_catalogue")
    tries = WordTries(manifest["vocabulary"]["words"])
    counters = Counter()
    started = time.monotonic()
    settings = dict(manifest["settings"])
    deadline = started + settings.pop("cooperative_budget_seconds")
    def render(left, right):
        text = " ".join(left+right)
        return text[:1].upper()+text[1:]+"."
    def allow_state(left, right):
        if len(left)+len(right) > job["max_words"]:
            counters["word_cap_rejections"] += 1
            return False
        return sum(map(len,left+right)) <= job["maximum"]
    def allow_closed(left, right):
        counters["closure_candidates"] += 1
        checks = mechanical_admission_checks(render(left,right),
                 min_letters=job["minimum"],max_letters=job["maximum"])
        admitted = all(checks[name] for name in basic_names)
        counters["basic_valid_closures"] += admitted
        return admitted
    common = dict(min_letters=job["minimum"], seed=job["seed"],
                  allow_state=allow_state,allow_closed=allow_closed,deadline=deadline)
    if job["method"] == "outside_in":
        settings.pop("max_overhang")  # existing outside-in implementation fixes this at 24
        words = beam_search(tries,ZipfScorer(),initial_state=State(0.0,(),(),"","L",0.0),
                            max_word_uses=None,**common,**settings)
    else:
        words = centerout_search(tries,ZipfScorer(),center="",**common,**settings)
    elapsed = time.monotonic()-started
    text = render(tuple(words),()) if words else None
    checks = mechanical_admission_checks(text,min_letters=job["minimum"],
               max_letters=job["maximum"]) if text else {}
    tape = "".join(c.lower() for c in text if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") if text else ""
    if text:
        assert tape == normalize_letters(text) == tape[::-1]
        assert job["minimum"] <= len(tape) <= job["maximum"]
        assert len(words) <= job["max_words"] and all(checks[n] for n in basic_names)
    return {**job,"status":"output" if text else "no_output", "deadline_reached":elapsed>=60,
            "elapsed_seconds":elapsed,"text":text,"letters":len(tape),"words":len(words),
            "normalized_sha256":hashlib.sha256(tape.encode()).hexdigest() if tape else None,
            "text_sha256":hashlib.sha256(text.encode()).hexdigest() if text else None,
            "independently_exact":bool(tape and tape==tape[::-1]),
            "strict_admitted":bool(text and all(checks.values())),"checks":checks,
            "counters":dict(counters),"readability_evidence":"not evaluated"}


def run(workers):
    assert platform.system()=="Linux", "run substantive search on the NUC"
    assert importlib.metadata.version("wordfreq")=="3.1.1"
    manifest=json.loads((OUT/"manifest.json").read_text())
    for p,h in manifest["source_sha256"].items():
        assert digest(ROOT/p)==h, f"frozen source changed: {p}"
    words=manifest["vocabulary"]["words"]
    assert hashlib.sha256(("\n".join(words)+"\n").encode()).hexdigest()==manifest["vocabulary"]["filtered_word_sha256"]
    output=OUT/"results.json"
    if output.exists():
        raise FileExistsError("results already exist")
    result=dict(manifest_sha256=digest(OUT/"manifest.json"),status="running",workers=workers,
                python=platform.python_version(),results=[])
    save(output,result)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_job,j,manifest):j for j in manifest["jobs"]}
        for future in as_completed(futures):
            job=futures[future]
            try:
                row=future.result()
            except Exception as exc:
                row={**job,"status":"error","error":f"{type(exc).__name__}: {exc}","text":None}
            result["results"].append(row)
            save(output,result)
            print(json.dumps({"completed":len(result["results"]),"job":row["id"],
                              "status":row["status"],"letters":row.get("letters")}),flush=True)
    assert {r["id"] for r in result["results"]}=={j["id"] for j in manifest["jobs"]}
    result["status"]="complete" if all(r["status"] != "error" for r in result["results"]) else "completed_with_errors"
    result["results"].sort(key=lambda r:r["id"])
    save(output,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","run"))
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    freeze() if args.action=="freeze" else run(args.workers)
