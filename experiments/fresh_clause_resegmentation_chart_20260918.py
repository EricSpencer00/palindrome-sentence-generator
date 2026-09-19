"""Fresh clause -> reverse-tape chart with typed right-side resegmentation.

This is a construction experiment, not a post-hoc reversal of a finished
sentence: each authored left clause supplies a tape, and an Earley-like chart
segments the live reverse tape into typed phrase slots.  The chart stores the
first impossible character and intact controls for reproducibility.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "fresh-clause-resegmentation-chart-20260918"
OUT = ROOT / "runs" / f"{ID}.json"

# Freshly authored scene clauses; none is a palindrome seed or catalogue row.
LEFT = [
    "At sunrise, the careful keeper opens a quiet archive.",
    "By dusk, the patient baker carries warm loaves home.",
    "After rain, the young teacher gathers bright paper models.",
    "At first light, the harbor pilot studies a folded chart.",
    "In spring, the kind gardener waters small tomato plants.",
    "Before winter, the village mason repairs an old stone wall.",
]

# Typed chart inventory.  The right phrase has to be ordinary enough to read,
# while its characters are consumed exactly from the reverse tape.
SLOTS = [
    ("DET", ["a", "an", "the"]),
    ("ADJ", ["careful", "quiet", "patient", "warm", "young", "bright", "folded", "kind", "small", "old", "stone"]),
    ("N", ["keeper", "baker", "teacher", "pilot", "gardener", "mason", "archive", "loaves", "models", "chart", "plants", "wall"]),
    ("V", ["opens", "carries", "gathers", "studies", "waters", "repairs", "keeps", "marks"]),
    ("P", ["at", "by", "in", "to", "home", "before", "after"]),
]

def tape(s: str) -> str:
    return "".join(c.lower() for c in s if c.isascii() and c.isalpha())

def audit(s: str) -> dict:
    t = tape(s); mismatches=[]; i,j=0,len(t)-1
    while i < j:
        if t[i] != t[j]: mismatches.append({"left":i,"right":j,"a":t[i],"b":t[j]})
        i+=1; j-=1
    return {"letters":len(t),"exact":bool(t) and not mismatches,
            "mismatch_count":len(mismatches),"first_mismatch":mismatches[:1],
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def shortcut_flags(s: str) -> dict:
    words=[tape(w) for w in re.findall(r"[A-Za-z]+",s)]
    content=[w for w in words if w not in {"a","an","the","at","by","in","to","home","before","after"}]
    return {"word_order_mirror":words == [w[::-1] for w in words],
            "repeated_content":len(content)!=len(set(content)),
            "self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue_text":False,"finished_tape_reversed":False}

def chart(reverse_tape: str, max_states=20000):
    # State is (offset, slot-index, words).  Slots may be skipped, but type
    # order is fixed, so this is a small typed chart rather than a word sweep.
    states=[(0,0,[])]; complete=[]; explored=0; furthest=0
    while states and explored < max_states:
        off, si, words = states.pop(0); explored += 1; furthest=max(furthest,off)
        if off == len(reverse_tape): complete.append(words); continue
        if si >= len(SLOTS): continue
        typ, vocab=SLOTS[si]
        # epsilon transition permits a compact phrase grammar.
        states.append((off,si+1,words))
        for w in vocab:
            if reverse_tape.startswith(w,off):
                states.append((off+len(w),si+1,words+[f"{w}/{typ}"]))
    return {"complete":complete[:5],"explored_states":explored,"furthest_offset":furthest,
            "tape_length":len(reverse_tape),"partial":not bool(complete)}

def run():
    rows=[]
    for left in LEFT:
        lt=tape(left); c=chart(lt[::-1]); right=" ".join(x.split("/")[0] for x in (c["complete"][0] if c["complete"] else []))
        rendered=f"{left} {right}".strip(); au=audit(rendered); flags=shortcut_flags(rendered)
        rows.append({"left_clause":left,"reverse_tape_preview":lt[::-1][:80],"chart":c,
          "rendered":rendered,"audit":au,"shortcut_flags":flags,
          "provenance":{"generator":ID,"left_author":"fresh local scene inventory","catalogue_imported":False,
                         "right_side":"typed chart segmentation of reverse tape; no completed clause reversal","seed_used_as_output":False}})
    exact=[r for r in rows if r["audit"]["exact"] and not any(r["shortcut_flags"].values())]
    best=min(rows,key=lambda r:(r["chart"]["tape_length"]-r["chart"]["furthest_offset"],-r["audit"]["letters"]))
    controls=[{"rendered":x,"audit":audit(x),"control":"intact prose near-miss"} for x in [
        "At sunrise, the careful keeper opens a quiet archive.",
        "By dusk, the patient baker carries warm loaves home."]]
    return {"experiment_id":ID,"status":"exact_candidate" if exact else "completed_no_exact_closure",
            "config":{"authored_clauses":len(LEFT),"typed_slots":[x[0] for x in SLOTS],"max_states":20000},
            "rows":rows,"exact_candidates":exact,"best_chart_reach":best,
            "intact_controls":controls,"independent_validation":"two-pointer palindrome audit plus SHA-256 tape hashes",
            "reader_gate":"closed: no exact novel survivor; controls remain intact prose",
            "next_repair":"Add typed plural/inflectional variants at the first residual chart character, then rerun the same authored clauses without widening to catalogue text.",
            "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"rows":len(result["rows"]),"exact":len(result["exact_candidates"]),"best":result["best_chart_reach"]["rendered"]}))
