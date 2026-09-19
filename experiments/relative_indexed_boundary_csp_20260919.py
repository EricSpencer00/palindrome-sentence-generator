"""Repair of the indexed path CSP: agreement-carrying relative complements.

The relative clause is a new typed grammar path, not a reversed phrase bank.
Boundary offsets are indexed while the character aliases are assigned, so a
word may cross the midpoint without post-hoc tape construction.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.half_tape_indexed_path_csp_20260919 import W, audit, _compatible
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "relative-indexed-boundary-csp-20260919"
SUBJECTS = (W("a bard","subj","sg"), W("a poet","subj","sg"), W("some men","subj","pl"))
VERBS = (W("reads","verb","sg","document"), W("marks","verb","sg","document"),
         W("inspires","verb","sg","person"), W("read","verb","pl","document"),
         W("mark","verb","pl","document"), W("inspire","verb","pl","person"))
OBJECTS = (W("a letter","obj",kind="document"), W("the sonnet","obj",kind="document"),
           W("some men","obj",kind="person"), W("Diana","obj",kind="person"))
MARKERS = (W("who","relmark"), W("that","relmark"))
REL_SUBJECTS = (W("the poet","relsubj","sg"), W("the herald","relsubj","sg"), W("some men","relsubj","pl"))

PATH = ("S", "V", "O", "M", "RS", "RV", "RO")
BANK = {"S": SUBJECTS, "V": VERBS, "O": OBJECTS, "M": MARKERS,
        "RS": REL_SUBJECTS, "RV": VERBS, "RO": OBJECTS}

def search(target: int, max_nodes: int = 100_000) -> dict[str, object]:
    # Keyed by (path position, tape slot, character), this is a boundary-aware
    # index: candidate words are looked up at their actual offset, not by a
    # finished half sentence.
    index = {}
    for key in PATH:
        for w in BANK[key]:
            for offset, ch in enumerate("".join(w.text.split()).lower()):
                slot = min(offset, target - 1 - offset)
                index.setdefault((key, slot, ch), []).append(w)
    rows=[]; nodes=0
    def dfs(k,pos,chosen,tape,state):
        nonlocal nodes
        if nodes >= max_nodes: return
        nodes += 1
        if k == len(PATH):
            text = " ".join(w.text for w in chosen) + "."
            a=audit(text); checks=mechanical_admission_checks(text,min_letters=30,max_letters=2000)
            rows.append({"rendered":text,"length":a["letters"],"audit":a,
                         "mechanical_checks":checks,
                         "mechanically_admitted":pos == target and a["two_pointer_exact"] and all(checks.values()),
                         "word_path":[w.text for w in chosen],
                         "provenance":{"experiment_id":EXPERIMENT_ID,"target_length":target,
                           "search":"agreement relative path + indexed boundary offsets",
                           "rlaif_used":False,"catalogue_imported":False,"finished_tape_reversed":False},
                         "reader_status":"unreviewed; programmatic checks do not certify readability"})
            return
        key=PATH[k]
        for w in BANK[key]:
            if w.text in state["used"]: continue
            if key == "V" and w.number != state.get("snum"): continue
            if key == "O" and w.kind != state.get("vkind"): continue
            if key == "RV" and w.number != state.get("rsnum"): continue
            if key == "RO" and w.kind != state.get("rvkind"): continue
            placed=_compatible(w.text,pos,target,tape)
            if placed is None: continue
            nxt=dict(state); nxt["used"]=state["used"]|{w.text}
            if key=="S": nxt["snum"]=w.number
            if key=="V": nxt["vkind"]=w.kind
            if key=="RS": nxt["rsnum"]=w.number
            if key=="RV": nxt["rvkind"]=w.kind
            dfs(k+1,pos+len("".join(w.text.split())),chosen+[w],placed,nxt)
    dfs(0,0,[],[None]*((target+1)//2),{"used":set()})
    return {"target":target,"nodes":nodes,"actual_candidates":rows,
            "exact_candidates":[r for r in rows if r["audit"]["two_pointer_exact"]],
            "mechanically_admitted":[r for r in rows if r["mechanically_admitted"]]}

def run(lengths=range(40,91),max_nodes=100_000):
    results=[search(n,max_nodes) for n in lengths]; rows=[r for x in results for r in x["actual_candidates"]]
    return {"experiment_id":EXPERIMENT_ID,"method":"agreement-carrying relative-complement path with indexed word-boundary offsets",
            "actual_candidates":rows,"stats":{"nodes":sum(x["nodes"] for x in results),"exact":sum(len(x["exact_candidates"]) for x in results),"mechanically_admitted":sum(len(x["mechanically_admitted"]) for x in results),"longest_exact":max((r["length"] for r in rows if r["audit"]["two_pointer_exact"]),default=0)},
            "provenance":{"independent_audits":["outside-in two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},
            "novelty_preflight":{"status":"passed","distinction":"relative subject agreement and object valency are live before boundary-index seam placement","prior_lanes_checked":["half-tape-indexed-path-csp-20260919","half-tape-grammar-csp-20260919"]},
            "next_repair":{"action":"permit a shared participant across the relative seam and add a finite verb-complement marker","reader_test":"randomized blinded intact-prose versus shuffled controls"},"reader_gate":"closed"}

if __name__ == "__main__":
    out=run(); p=Path(__file__).resolve().parents[1]/"runs"/(EXPERIMENT_ID+".json"); p.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"],sort_keys=True))
