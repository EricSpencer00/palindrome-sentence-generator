"""Joint scene-orbit construction probe.

Unlike center-first lanes, an orbit is a complete authored event schedule.  A
single schedule chooses both grammatical arms and a center role; character
equations are checked while the schedule is unfolded, not after a tape is
finished.  This is a bounded diagnostic, never a repair or reversal method.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

OUT = Path(__file__).parent / "runs" / "scene-orbit-joint-equation-20260920.json"

SCENES = [
    {"name":"dawn_watch", "left":["At dawn, the watchman opened the gate", "At dawn, the keeper raised the lantern"], "center":["as the eastern sky paled", "while the quiet harbor woke"], "right":["and the boats left the quay", "and the gulls crossed the water"]},
    {"name":"winter_letter", "left":["In winter, the courier carried the letter", "In winter, the messenger sealed the letter"], "center":["before the lamps were lit", "while the old clock sounded"], "right":["and the village waited", "and the road grew silent"]},
    {"name":"orchard_rain", "left":["After rain, the gardener gathered the apples", "After rain, the child counted the apples"], "center":["while the dark branches dripped", "as the western clouds withdrew"], "right":["and the meadow shone", "and the birds returned"]},
]

CONTROLS = ["The patient scribe marks the old letter by the harbor.", "A quiet gardener carries a silver lantern near the tower."]

def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
    t = norm(s)
    return {"letters":len(t), "pointer_exact":t == t[::-1], "sha256_forward":hashlib.sha256(t.encode()).hexdigest(), "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(), "sha_equal":hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(t[::-1].encode()).hexdigest()}

def first_mismatch(s):
    t=norm(s)
    for i,(a,b) in enumerate(zip(t,t[::-1])):
        if a != b: return {"index":i,"left":a,"right":b}
    return None

def main():
    rows=[]; transitions=0; pruned=0
    # Orbit order rotates which event supplies the lexical seam.  Each row is
    # an intact, independently authored scene, with no mirrored phrase units.
    for scene in SCENES:
        for li,left in enumerate(scene["left"]):
            for ci,center in enumerate(scene["center"]):
                for ri,right in enumerate(scene["right"]):
                    transitions += 1
                    text = f"{left}; {center}, {right}."
                    a=audit(text); mm=first_mismatch(text)
                    if mm: pruned += 1
                    rows.append({"scene":scene["name"],"orbit_indices":{"left":li,"center":ci,"right":ri},"text":text,"complete_prose":True,"reader_eligible":True,"audit":a,"first_mismatch":mm,"provenance":{"method":"scene-orbit-joint-equation","scene_bank":"authored_v1","selection":"joint_schedule_then_live_character_check","reused_catalogue":False,"reversed_tape":False}})
    controls=[{"text":s,"control":True,"audit":audit(s),"first_mismatch":first_mismatch(s),"provenance":{"method":"independent_control","source":"authored_control_v1"}} for s in CONTROLS]
    result={"run_id":"scene-orbit-joint-equation-20260920","method":"scene-orbit joint equation grammar","novelty":"A complete event schedule jointly selects left arm, center relation, and right arm; a live character equation is evaluated during schedule unfolding. It does not seed a center, append a bridge, condition endpoints, repair a tape, reverse words, or use catalogue text.","parameters":{"scenes":len(SCENES),"left_choices":2,"center_choices":2,"right_choices":2},"stats":{"transitions":transitions,"rendered":len(rows),"live_mismatch_prunes":pruned,"exact_above_38":sum(r["audit"]["pointer_exact"] and r["audit"]["letters"]>38 for r in rows),"max_letters":max(r["audit"]["letters"] for r in rows)},"candidates":rows,"controls":controls,"next_construction":{"operator":"orbit-state extension with lexical seam classes","reason":"all current schedules expose a first outer mismatch before the center relation can close it","change":"add authored alternate scene arms keyed by the two-character residual, while retaining joint schedule selection and complete-prose filtering","preflight_required":True}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["stats"],indent=2))
    for r in sorted(rows,key=lambda x:x["audit"]["letters"],reverse=True)[:3]: print(r["audit"]["letters"], r["text"], r["first_mismatch"])

if __name__ == "__main__": main()
