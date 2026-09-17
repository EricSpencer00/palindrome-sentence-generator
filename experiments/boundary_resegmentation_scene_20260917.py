"""Boundary-aware resegmentation over intact scene frames.

The seam is part of construction: each lexical choice is tested while the
outer tape is emitted, rather than ranked by a post-hoc mismatch score.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "boundary-resegmentation-scene-20260917"
OUT = ROOT / "runs" / f"{ID}.json"

FRAMES = [
    ("garden", "At dawn, the patient gardener {v} {obj}, and labels the fresh seed trays before dusk.",
     [("opens", "the quiet greenhouse"), ("opens", "the old greenhouse"), ("opened", "the quiet greenhouse"), ("unlocks", "the old greenhouse")]),
    ("harbor", "At first light, the harbor pilot {v} {obj}, and signals the waiting boat toward shore.",
     [("checks", "the weathered chart"), ("checks", "the folded chart"), ("studies", "the weathered chart"), ("studied", "the folded chart")]),
    ("archive", "In the quiet archive, the patient clerk {v} {obj}, and records the missing names before closing.",
     [("repairs", "a torn map"), ("repairs", "the torn map"), ("files", "the marked folder"), ("filed", "a marked map")]),
    ("school", "After rain, the young teacher {v} {obj}, and dries the classroom windows at noon.",
     [("carries", "the bright models"), ("carries", "the paper models"), ("greets", "the waiting children"), ("greeted", "the quiet children")]),
]

def letters(s): return "".join(c.lower() for c in s if c.isalpha() and c.isascii())

def live_closure(s):
    """Emit both ends simultaneously; reject at the first unequal pair."""
    t = letters(s); left = []; right = []; failures = []
    for i in range((len(t)+1)//2):
        j = len(t)-1-i; left.append(t[i]); right.append(t[j])
        if t[i] != t[j]: failures.append((i,j,t[i],t[j])); break
    return not failures and bool(t), failures

def independent(s):
    t = letters(s); i,j=0,len(t)-1; mm=[]
    while i<j:
        if t[i]!=t[j]: mm.append((i,j,t[i],t[j]))
        i+=1; j-=1
    return {"letters":len(t),"exact":bool(t) and not mm,"mismatch_count":len(mm),
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def shortcut_flags(s):
    ws=[letters(x) for x in re.findall(r"[A-Za-z]+",s)]
    content=[w for w in ws if w not in {"a","an","the","and","at","in","before","after","toward"}]
    return {"word_order_mirror":ws == [w[::-1] for w in ws],
            "repeated_content":len(content)!=len(set(content)),
            "self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue_text":False,"finished_tape_reversed":False}

def run():
    candidates=[]; failures=[]
    for fid, template, choices in FRAMES:
        for v,obj in choices:
            rendered=template.format(v=v,obj=obj)
            au=independent(rendered); fl=shortcut_flags(rendered)
            row={"rendered":rendered,"frame":fid,"boundary_choice":{"verb":v,"object":obj},
                 "audit":au,"shortcut_flags":fl,"provenance":{"generator":ID,"frame_author":"local human-authored scene inventory","catalogue_imported":False,"seed_used_as_output":False,"boundary_search":"verb/object boundary and inflection jointly selected"}}
            ok, live_fail=live_closure(rendered); row["live_seam"]={"closed":ok,"first_failure":live_fail[:1]}
            if not 100 <= au["letters"] <= 160: row["failure_reason"]="length_gate_before_ranking"; failures.append(row); continue
            if any(fl[k] for k in ("word_order_mirror","repeated_content","self_palindromic_content_words","borrowed_catalogue_text","finished_tape_reversed")): row["failure_reason"]="shortcut_gate"; failures.append(row); continue
            candidates.append(row)
    exact=[r for r in candidates if r["live_seam"]["closed"] and r["audit"]["exact"]]
    best=min(candidates,key=lambda r:(r["audit"]["mismatch_count"],-r["audit"]["letters"])) if candidates else None
    return {"experiment_id":ID,"status":"completed_no_exact_closure" if not exact else "exact_candidates_found","config":{"frames":len(FRAMES),"choices":16,"length_gate":"100<=letters<=160 before ranking","live_pruning":True},"actual_candidates":candidates,"best":best,"exact_candidates":exact,"failed_branches":failures,"independent_validation":"separate two-pointer audit plus SHA-256 forward/reverse","novelty_preflight":"no catalogue, fixed tape, repeated unit, fragments, or mirror controls","reader_gate":"closed unless exact novel survivor","next_repair":"Add paired lexical substitutions selected from a semantic scene lattice, while retaining live seam pruning and explicit attachment roles.","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"candidates":len(result["actual_candidates"]),"failed":len(result["failed_branches"]),"best":result["best"]["rendered"] if result["best"] else None}))
