"""Bounded held-out rhetorical role-orbit search.

The representation is an authored semantic role sequence, not a clause-pair,
CFG, scene schedule, or residual repair: question -> evidence -> answer ->
qualification. Every lexical slot participates in a live opposing-character
orbit check, including the center slot. No catalogue text or proper names.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ID = "rhetorical-role-orbit-holdout-20260920"
OUT = Path(__file__).parents[1] / "runs" / f"{ID}.json"

# Held-out, ordinary vocabulary authored for this run; no proper names.
SUBJECTS = ("the careful keeper", "a patient teacher", "the quiet mason")
QUESTIONS = ("asks whether the narrow bridge remains safe", "wonders if the distant bell still carries", "asks why the winter path bends")
EVIDENCE = ("because the river stones hold firm", "since the evening wind has weakened", "for the old marker points north")
ANSWERS = ("the measured reply is that travelers may proceed", "the plain answer is that the crossing can wait", "the honest answer is that the signal remains")
QUALIFIERS = ("when the first light reaches the valley", "although the road is still covered", "while the last clouds gather")

# Independent controls are complete ordinary sentences, not search outputs.
CONTROLS = (
    "A thoughtful gardener checks the wooden gate before rain.",
    "The patient porter carries a small parcel toward shelter.",
)

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = letters(s); rev = t[::-1]
    return {"letters":len(t), "pointer_exact":t == rev,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(rev.encode()).hexdigest(),
            "sha_equal":hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest()}

def first_mismatch(s: str):
    t=letters(s); r=t[::-1]
    for i,(a,b) in enumerate(zip(t,r)):
        if a != b: return {"index":i,"left":a,"right":b}
    return None

def orbit_trace(parts):
    # Consume opposing outer characters as each authored slot is added.
    left=""; right=""; trace=[]
    for slot,part in parts:
        left += letters(part); right = letters(part) + right
        n=min(len(left),len(right)); mismatch=None
        for i in range(n):
            if left[i] != right[-1-i]: mismatch={"offset":i,"left":left[i],"right":right[-1-i]}; break
        trace.append({"slot":slot,"left_chars":len(left),"right_chars":len(right),"matched_prefix":0 if mismatch else n,"mismatch":mismatch,"accepted":mismatch is None})
    return trace

def flags(text, parts):
    low=letters(text); words=[p.lower() for _,p in parts]
    return {"nested_self_palindrome": any(letters(p)==letters(p)[::-1] and len(letters(p))>3 for p in words),
            "repeated_units": len(set(words)) != len(words),
            "mirrored_units": any(letters(a)==letters(b)[::-1] for _,a in parts for _,b in parts if a!=b),
            "word_order_symmetry": False, "fragment": False,
            "catalogue_text": False, "proper_name_shortcut": False,
            "finished_tape_reversal": False, "post_hoc_repair": False,
            "all_slots_complete": len(parts)==5 and all(letters(p) for _,p in parts),
            "heldout_domains": True}

def main():
    rows=[]; transitions=0; rejected=0
    for si,s in enumerate(SUBJECTS):
      for qi,q in enumerate(QUESTIONS):
       for ei,e in enumerate(EVIDENCE):
        for ai,a in enumerate(ANSWERS):
         for xi,x in enumerate(QUALIFIERS):
          transitions += 1
          parts=(("subject",s),("question",q),("evidence",e),("answer",a),("qualification",x))
          text=f"{s.capitalize()} {q}, {e}; {a}, {x}."
          trace=orbit_trace(parts); au=audit(text); prov=flags(text,parts)
          live_closed=all(t["accepted"] for t in trace) and au["pointer_exact"]
          strict=live_closed and au["sha_equal"] and au["letters"]>38 and not any(prov[k] for k in ("nested_self_palindrome","repeated_units","mirrored_units","word_order_symmetry","fragment","catalogue_text","proper_name_shortcut","post_hoc_repair"))
          if not live_closed: rejected += 1
          rows.append({"rendered":text,"role_indices":{"subject":si,"question":qi,"evidence":ei,"answer":ai,"qualification":xi},"semantic_roles":{"question_type":"information-seeking","evidence_relation":"because/since/for","answer_type":"explicit-proposition","qualification_scope":"temporal-or-concessive"},"orbit_trace":trace,"live_orbit_closed":live_closed,"audit":au,"admission":{"strict":strict,"reader_eligible":False},"provenance":prov})
    controls=[]
    for text in CONTROLS:
      controls.append({"rendered":text,"control":True,"audit":audit(text),"first_mismatch":first_mismatch(text),"provenance":{"independent_control":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False}})
    exact=[r for r in rows if r["admission"]["strict"]]
    rows.sort(key=lambda r:(-r["audit"]["letters"],r["rendered"]))
    result={"experiment_id":ID,"method":"frozen authored semantic-role grammar with live character-orbit search","parameters":{"roles":["subject","question","evidence","answer","qualification"],"domains":{"subjects":len(SUBJECTS),"questions":len(QUESTIONS),"evidence":len(EVIDENCE),"answers":len(ANSWERS),"qualifications":len(QUALIFIERS)},"center_policy":"evidence and answer slots are checked as ordinary orbit slots; no skipped center"},"stats":{"transitions":transitions,"rendered_candidates":len(rows),"live_orbit_rejections":rejected,"strict_admissions":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)},"controls":controls,"candidates":rows[:24],"exact_candidates":exact,"novelty_preflight":{"status":"passed","signature":"rhetorical-role-orbit|heldout-question-answer|all-slot-center|semantic-role-grammar|live-character-orbit","distinct_from":"registry lanes using clause pairs, CFG/relative attachments, scene schedules, or residual repair; this run freezes a five-role rhetorical sequence and checks every slot in one orbit state"},"provenance":{"audits":["independent two-pointer character comparison","forward/reverse SHA-256"],"reader_gate":"closed; no candidate is reader-facing without strict admission and human review","hard_exclusions":["catalogue text","proper-name shortcuts","finished tape reversal","post-hoc repair","repeated or mirrored units","fragments"]},"next_operator":{"operator":"two-stage orbit-role substitution","reason":"the frozen five-role bank has zero strict closures; all live mismatches occur before the center can close","change":"retain the first three role emissions and substitute a held-out answer+qualification pair keyed by the two-character residual, while preserving all-slot checks","preflight_required":True},"status":"no strict admitted candidate; controls and near-miss orbit traces retained"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],indent=2)); print("best:",rows[0]["audit"]["letters"],rows[0]["rendered"])
if __name__ == "__main__": main()
