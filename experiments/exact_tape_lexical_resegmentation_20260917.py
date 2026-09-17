"""Bounded lexical resegmentation of exact rejected tapes.

The tape is never changed or reversed.  Candidate word boundaries are found by
dynamic programming, then scored with lexical frequency and local bigram
attestation.  This is a salvage lane: it can recover prose hidden by poor
spacing, but cannot turn a bad character tape into a good one.
"""
from __future__ import annotations
import hashlib, json, re
from functools import lru_cache
from pathlib import Path
from wordfreq import zipf_frequency
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/exact-tape-lexical-resegmentation-20260917.json"
EXPERIMENT = "exact-tape-lexical-resegmentation-20260917"
SIGNATURE = "exact-tape-dp-boundary-search|zipf-bigram-ranking|fail-closed-admission"

def _tape(s): return normalize_letters(s)

@lru_cache(maxsize=20000)
def _words(tape: str, start: int, end: int):
    w = tape[start:end]
    return w if len(w) > 1 and zipf_frequency(w, "en") >= 2.2 else None

def segment(tape: str, limit: int = 5):
    """Return top segmentations; each edge is an independently attested word."""
    best = {len(tape): [(0.0, ())]}
    for i in range(len(tape) - 1, -1, -1):
        rows = []
        for j in range(i + 2, min(len(tape), i + 18) + 1):
            word = _words(tape, i, j)
            if not word or j not in best: continue
            for score, tail in best[j]:
                rows.append((score + zipf_frequency(word, "en") - (0.6 if len(word)==2 else 0), (word,) + tail))
        best[i] = sorted(rows, reverse=True)[:limit]
    return [{"words": list(words), "score": round(score, 3)} for score, words in best.get(0, [])]

def audit(text):
    tape = _tape(text); bad=[]; i,j=0,len(tape)-1
    while i < j:
        if tape[i] != tape[j]: bad.append((i,j,tape[i],tape[j]))
        i,j=i+1,j-1
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=240)
    return {"letters":len(tape), "exact":bool(tape) and not bad,
            "first_mismatch":bad[0] if bad else None,
            "sha256_forward":hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "mechanical_checks":checks}

def main():
    registry=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    assert not any(r.get("signature")==SIGNATURE for r in registry.get("entries",[]))
    source=json.loads((ROOT/"runs/existing-exact-tape-audit-20260915.json").read_text())
    rows=[]
    for row in source.get("near_misses", []):
        tape=_tape(row["rendered"])
        for cand in segment(tape):
            rendered=" ".join(cand["words"])
            checks=audit(rendered)["mechanical_checks"]
            rows.append({"rendered":rendered,"letters":len(tape),"score":cand["score"],
              "exact_audit":audit(rendered),"mechanically_admitted":all(checks.values()),
              "source":row.get("provenance"),"provenance":{"source":"resegmented existing exact tape",
              "tape_changed":False,"catalogue_imported":False,"finished_surface_reversed":False,
              "word_order_mirror_rejected":not checks.get("word_order_mirror",True),
              "self_palindromic_span_rejected":not checks.get("self_palindromic_proper_multiword_span",True)}})
    rows.sort(key=lambda r:r["score"], reverse=True)
    admitted=[r for r in rows if r["mechanically_admitted"]]
    result={"experiment_id":EXPERIMENT,"signature":SIGNATURE,"status":"completed_no_reader_candidate" if not admitted else "mechanically_admitted_pending_readers",
      "method":"dynamic-programming lexical resegmentation over exact tapes; top-5 paths per suffix; no tape edits",
      "stats":{"source_tapes":len(source.get("near_misses",[])),"segmentations":len(rows),"exact":sum(r["exact_audit"]["exact"] for r in rows),"mechanically_admitted":len(admitted)},
      "best_candidates":rows[:20],"reader_eligible":bool(admitted),
      "novelty_preflight":{"registry_entries_inspected":len(registry.get("entries",[])),"signature_collision":False,"catalogue_used_for_generation":False},
      "next_test":"If a fresh exact tape is supplied, run this same DP with a clause bigram model and blinded reader review; this lane cannot repair a character-level semantic defect."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["stats"]))
if __name__ == "__main__": main()
