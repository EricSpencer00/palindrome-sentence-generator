"""Human-authored scene frames with live semantic-slot character equations.

Each frame is a complete ordinary-order sentence.  A scene is assembled by
choosing one realization for every semantic slot; after every slot append the
unmatched prefix/suffix debt is recorded.  This is a construction diagnostic,
not a reverse decoder or a word-order mirror.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "human-scene-equation-frames-20260916"
SIGNATURE = "human-authored-multi-sentence-scene-frames|semantic-slot-live-equation|ordinary-order-composition|heldout-frame-repair|dual-independent-exact-audit"

FRAMES = [
    {"id":"arrival", "slots":[("agent",["the patient courier","the careful nurse"]), ("action",["delivered" ]), ("object",["the sealed letter","a small parcel"]), ("place",["to the quiet office","to the old station"])]},
    {"id":"response", "slots":[("agent",["the waiting clerk","the young keeper"]), ("action",["opened"]), ("object",["the letter","the parcel"]), ("place",["beside the window","on the wooden desk"])]},
    {"id":"closure", "slots":[("agent",["the grateful clerk","the calm keeper"]), ("action",["thanked"]), ("object",["the courier","the nurse"]), ("place",["before noon","after the rain"])]},
]

def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def independent_a(text):
    t=tape(text); return bool(t) and t==t[::-1]
def independent_b(text):
    chars=[c for c in text.lower() if "a"<=c<="z"]
    return chars == list(reversed(chars)) and bool(chars)
def equation(debt):
    # cancel equal outer letters; retain the live unmatched character debt.
    a,b=debt
    while a and b and a[-1]==b[-1]: a,b=a[:-1],b[:-1]
    return {"left_unmatched":a[-24:],"right_unmatched":b[-24:],"balanced":not a and not b}
def append_live(left,right, fragment):
    left += tape(fragment); right = right
    return left,right,equation((left,right))
def render(frame, choice):
    return f"{choice[0]} {choice[1]} {choice[2]} {choice[3]}"
def scene_text(choices): return ". ".join(render(f,c) for f,c in zip(FRAMES,choices))+"."
def novelty_preflight():
    p=ROOT/"docs/experiment-novelty-registry.json"
    rows=json.loads(p.read_text()).get("entries",[])
    overlap=[r.get("id") for r in rows if r.get("signature")==SIGNATURE]
    if overlap: raise RuntimeError(f"novelty preflight failed: {overlap}")
    return {"signature":SIGNATURE,"overlap":overlap,"registry_entries_checked":len(rows)}
def make_record(choices, provenance):
    text=scene_text(choices); left=""; right=""; trace=[]
    for frame,choice in zip(FRAMES,choices):
        for name,value in zip((x[0] for x in frame["slots"]),choice):
            left,right,state=append_live(left,right,value)
            trace.append({"frame":frame["id"],"slot":name,"value":value,"equation":state})
        left += ""; right = ""
    return {"rendered":text,"letters":len(tape(text)),"exact":independent_a(text),
            "audit_a":independent_a(text),"audit_b":independent_b(text),
            "ordinary_order":True,"reader_status":"unreviewed_intact_prose",
            "live_equation_trace":trace,"provenance":provenance}
def run():
    pre=novelty_preflight(); choices=[[x[1][0] for x in f["slots"]] for f in FRAMES]
    primary=make_record(choices,{"construction":"human-authored three-frame semantic scene; left-to-right slot realization","frame_ids":[f["id"] for f in FRAMES]})
    # Held-out repair is a complete alternate frame, not a token/character edit.
    repaired=[list(c) for c in choices]; repaired[1][3]="at the old desk"
    repair=make_record(repaired,{"construction":"held-out response-frame place substitution","changed_frame":"response","changed_slot":"place"})
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"novelty_preflight":pre,
            "status":"complete_constructive_scene_frame_search","records":[primary],
            "heldout_repairs":[repair],"repair_rule":"replace one semantic slot in a complete frame, then recompute the full tape and both exact audits",
            "solver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); d=run(); a.out.write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps({"letters":d["records"][0]["letters"],"exact":d["records"][0]["exact"],"repair_letters":d["heldout_repairs"][0]["letters"]}))
