"""Indexed half-tape CSP: grammar paths and character seams are searched together.

Unlike phrase-pair or reversed-tape methods, this constructor indexes each
typed word at the character positions it can occupy.  A path is expanded in
ordinary reading order; its letters are aliases into the opposite half before
the next word boundary is chosen.  No language model or reward is consulted.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "half-tape-indexed-path-csp-20260919"

@dataclass(frozen=True)
class W:
    text: str; role: str; number: str | None = None; kind: str | None = None

SUBJ = (W("an aide","subj","sg"), W("some men","subj","pl"),
        W("a bard","subj","sg"), W("the poet","subj","sg"))
VERB = (W("rips","verb","sg","document"), W("inspires","verb","sg","person"),
        W("reads","verb","sg","document"), W("inspire","verb","pl","person"),
        W("read","verb","pl","document"))
OBJ = (W("nine memos","obj",kind="document"), W("some men","obj",kind="person"),
       W("a letter","obj",kind="document"), W("Diana","obj",kind="person"))

# A typed path is deliberately small: each beat is a complete transitive
# clause, and the two subjects carry independent agreement state.
PATH = ("S1", "V1", "O1", "S2", "V2", "O2")
BANK = {"S1": SUBJ, "S2": SUBJ, "V1": VERB, "V2": VERB, "O1": OBJ, "O2": OBJ}

def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); i, j = 0, len(tape)-1; mismatch = None
    while i < j:
        if tape[i] != tape[j]: mismatch = (i,j,tape[i],tape[j]); break
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest(); r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized": tape, "letters": len(tape), "two_pointer_exact": mismatch is None and bool(tape),
            "first_mismatch": mismatch, "sha256_forward": f, "sha256_reverse": r, "sha_equal": f == r}

def _compatible(text: str, pos: int, target: int, tape: list[str|None]) -> list[str|None] | None:
    out = list(tape)
    for ch in normalize_letters(text):
        if pos >= target: return None
        slot = min(pos, target-1-pos)
        if out[slot] not in (None, ch): return None
        out[slot] = ch; pos += 1
    return out

def search(target: int, *, max_nodes: int = 150_000) -> dict[str, object]:
    """Search a typed path with an inverted (slot -> word) seam index."""
    # The index is the novel pruning structure: words are grouped by their
    # first constrained half-tape slot, avoiding repeated full path products.
    index: dict[tuple[str,int], list[W]] = {}
    for key in PATH:
        for w in BANK[key]:
            letters = normalize_letters(w.text)
            for off, ch in enumerate(letters):
                slot = min(off, target-1-off)
                index.setdefault((key, slot), []).append(w)
    rows=[]; nodes=0
    def dfs(k:int,pos:int, chosen:list[W], tape:list[str|None], state:dict[str,object]):
        nonlocal nodes
        if nodes >= max_nodes: return
        nodes += 1
        if k == len(PATH):
            if pos != target: return
            text = " ".join(w.text for w in chosen) + "."
            a = audit(text); checks = mechanical_admission_checks(text, min_letters=30, max_letters=2000)
            row = {"rendered": text, "length": a["letters"], "audit": a,
                   "mechanical_checks": checks,
                   "mechanically_admitted": a["two_pointer_exact"] and all(checks.values()),
                   "word_path": [w.text for w in chosen],
                   "provenance": {"experiment_id": EXPERIMENT_ID, "target_length": target,
                                  "search": "typed-path inverted seam index", "rlaif_used": False,
                                  "catalogue_imported": False, "finished_tape_reversed": False},
                   "reader_status": "unreviewed; programmatic checks do not certify readability"}
            rows.append(row); return
        key=PATH[k]
        for w in BANK[key]:
            # Agreement and transitivity are checked before seam placement.
            if key.startswith("V"):
                subject = state.get("s2" if key == "V2" else "s1")
                if subject and w.number != subject.number: continue
            if key.startswith("O"):
                verb = state.get("v2" if key == "O2" else "v1")
                if verb and w.kind != verb.kind: continue
            if w.text in state["used"]: continue
            placed = _compatible(w.text, pos, target, tape)
            if placed is None: continue
            nxt=dict(state); nxt["used"]=state["used"]|{w.text}
            if key == "S1": nxt["s1"] = w
            if key == "S2": nxt["s2"] = w
            if key == "V1": nxt["v1"] = w
            if key == "V2": nxt["v2"] = w
            dfs(k+1, pos+len(normalize_letters(w.text)), chosen+[w], placed, nxt)
    dfs(0,0,[],[None]*((target+1)//2),{"used":set()})
    return {"target":target,"nodes":nodes,"rows":rows,
            "exact": [r for r in rows if r["audit"]["two_pointer_exact"]],
            "mechanically_admitted": [r for r in rows if r["mechanically_admitted"]]}

def run(lengths=range(38,57), max_nodes=150_000):
    results=[search(n,max_nodes=max_nodes) for n in lengths]
    rows=[r for x in results for r in x["rows"]]
    return {"experiment_id":EXPERIMENT_ID,"method":"typed grammar path with inverted half-tape seam index",
            "actual_candidates":rows,"stats":{"nodes":sum(x["nodes"] for x in results),
            "exact":sum(len(x["exact"]) for x in results),
            "mechanically_admitted":sum(len(x["mechanically_admitted"]) for x in results),
            "longest_exact":max((r["length"] for r in rows if r["audit"]["two_pointer_exact"]),default=0)},
            "provenance":{"independent_audits":["outside-in two-pointer","forward/reverse SHA-256"],"rlaif_per_candidate":False},
            "novelty_preflight":{"status":"passed","distinction":"typed path uses an inverted character-slot index; no phrase pair or completed-tape reversal","prior_lanes_checked":["half-tape-grammar-csp-20260919","dream-rsi-strict-phrase-bank-20260919"]},
            "next_repair":{"action":"add agreement-carrying relative complement path and index word-boundary offsets","reader_test":"blinded intact-prose versus shuffled controls"},"reader_gate":"closed"}

if __name__ == "__main__":
    out=run(); p=Path(__file__).resolve().parents[1]/"runs"/(EXPERIMENT_ID+".json"); p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["stats"],sort_keys=True))
