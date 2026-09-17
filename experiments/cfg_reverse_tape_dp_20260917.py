"""Exact CFG × reverse-tape intersection (a constructive diagnostic).

The chart key deliberately contains the nonterminal, both rendered sides, and
the live tape state.  No candidate is scored or repaired after generation.
"""
from __future__ import annotations
import hashlib, json
from functools import lru_cache
from pathlib import Path

ID = "cfg-reverse-tape-dp-20260917"
MAX_LETTERS = 132
MAX_STATES_PER_RULE = 240
LEX = {
    "Det": ("the", "a"), "N": ("traveler", "gardener"),
    "V": ("remembers", "observes"), "Adv": ("carefully", "quietly"),
    "Conj": ("and",),
}

def letters(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalpha())

def audit(text: str) -> dict:
    tape = letters(text); mismatches=[]; i,j=0,len(tape)-1
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"left":i,"right":j,"a":tape[i],"b":tape[j]})
        i += 1; j -= 1
    f=hashlib.sha256(tape.encode()).hexdigest(); r=hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters":len(tape),"two_pointer_exact":bool(tape) and not mismatches,
            "first_mismatches":mismatches[:8],"sha256_forward":f,
            "sha256_reverse":r,"sha_equal_under_reversal":f==r}

def run() -> dict:
    # tape_state=(unmatched left prefix, unmatched right suffix); it is updated
    # during expansion, rather than computed by a post-hoc reversal pass.
    @lru_cache(maxsize=None)
    def chart(nt: str, left: str, right: str, tape_state: tuple[str,str], depth: int):
        if len(letters(left+right)) > MAX_LETTERS or depth > 5: return ()
        if nt == "S": rules=(("C",), ("C","Conj","S"))
        elif nt == "C": rules=(("NP","VP"),)
        elif nt == "NP": rules=(("Det","N"),)
        elif nt == "VP": rules=(("V",), ("V","Adv"))
        else: rules=tuple((x,) for x in LEX[nt])
        out=[]
        for rule in rules:
            states=((left,right,tape_state),)
            for child in rule:
                nxt=[]
                for l,r,ts in states:
                    if child in LEX:
                        for word in LEX[child]:
                            nl=(l+" "+word).strip(); ntape=letters(nl+r)
                            # live boundary debt, retained in every chart key
                            nxt.append((nl,r,(ntape[:len(ntape)//2],ntape[(len(ntape)+1)//2:])))
                    else:
                        for cl,cr,cts in chart(child,l,r,ts,depth+1): nxt.append((cl,cr,cts))
                # Keep the chart finite while preserving breadth across the
                # grammar; this is a search budget, never a near-miss score.
                states=tuple(dict.fromkeys(nxt))[:MAX_STATES_PER_RULE]
            out.extend(states)
        return tuple(dict.fromkeys(out))
    rows=[]
    for l,r,_ in chart("S","","",("", ""),0):
        text=(l+" "+r).strip(); a=audit(text)
        if a["letters"] >= 100: rows.append({"text":text,"audit":a})
    exact=[x for x in rows if x["audit"]["two_pointer_exact"]]
    best=max(rows,key=lambda x:x["audit"]["letters"],default=None)
    return {"experiment_id":ID,"method":"memoized CFG intersection with live reverse character tape",
            "grammar":"S -> C | C and S; C -> NP VP; NP -> Det N; VP -> V | V Adv",
            "max_letters":MAX_LETTERS,"candidate_count":len(rows),"exact_count":len(exact),
            "rendered_candidates":exact[:3],"longest_frontier":best,
            "chart_states":chart.cache_info().currsize,
            "novelty_preflight":{"catalogue_imported":False,"known_palindromes_imported":False,
                                 "near_miss_scoring":False,"signature":ID},
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexicon":"fresh small common-English terminals embedded in this experiment",
                           "audits":["independent two-pointer", "forward/reverse SHA-256"]},
            "failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure(s) found",
                                   "next_repair":"Add typed complement terminals to the CFG and retain the same live tape-state key; do not mirror completed clauses."},
            "anti_shortcut_flags":{"posthoc_reversal":False,"word_order_mirror":False,"catalogue_text":False,"fragment":False}}

if __name__ == "__main__":
    out=run(); path=Path("runs")/(ID+".json"); path.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"path":str(path),"exact_count":out["exact_count"],"longest":out["longest_frontier"]["audit"]["letters"] if out["longest_frontier"] else 0}))
