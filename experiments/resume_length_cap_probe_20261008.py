"""Resume uncheckpointed jobs after connection loss, preserving first-attempt data.

This changes orchestration only; the frozen per-job search function and inputs
are hash-verified. Uncheckpointed attempts have unknown outcomes and are
explicitly reported as interrupted, rather than counted as search failures.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.length_cap_probe_20261008 import OUT, ROOT, digest, run_job, save


def main():
    manifest=json.loads((OUT/"manifest.json").read_text())
    for p,h in manifest["source_sha256"].items():
        assert digest(ROOT/p)==h, f"frozen source changed: {p}"
    initial=OUT/"results-interrupted-1.json"
    if initial.exists():
        raise FileExistsError("continuation already initiated")
    old=OUT/"results.json"
    previous=json.loads(old.read_text())
    assert previous["status"]=="running"
    assert previous["manifest_sha256"]==digest(OUT/"manifest.json")
    done={r["id"] for r in previous["results"]}
    assert len(done)==len(previous["results"])
    missing=[j for j in manifest["jobs"] if j["id"] not in done]
    old.rename(initial)
    continuation=dict(started_utc=datetime.now(timezone.utc).isoformat(),
                      runner_sha256=digest(__file__),initial_results_sha256=digest(initial),
                      completed_before_interruption=sorted(done),restarted_jobs=[j["id"] for j in missing],
                      interruption="SSH connection ended; no live runner remained; outcomes of uncheckpointed attempts unknown")
    result={**previous,"continuation":continuation,"status":"running"}
    save(old,result)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_job,j,manifest):j for j in missing}
        for future in as_completed(futures):
            j=futures[future]
            try:
                row=future.result()
            except Exception as exc:
                row={**j,"status":"error","text":None,"error":f"{type(exc).__name__}: {exc}"}
            row["resumed_after_interruption"]=True
            result["results"].append(row)
            save(old,result)
            print(json.dumps({"completed":len(result["results"]),"job":row["id"],
                              "status":row["status"],"letters":row.get("letters")}),flush=True)
    assert {r["id"] for r in result["results"]}=={j["id"] for j in manifest["jobs"]}
    result["status"]="complete" if all(r["status"]!="error" for r in result["results"]) else "completed_with_errors"
    result["results"].sort(key=lambda r:r["id"])
    result["completed_utc"]=datetime.now(timezone.utc).isoformat()
    save(old,result)


if __name__=="__main__":
    main()
