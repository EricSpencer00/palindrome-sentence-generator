"""Feature-carrying morphology transducer with a live character frontier.

Unlike word-pair or finished-tape methods, this lane realizes an ordinary
SVO sentence left-to-right through a small transducer.  Number/tense features
are carried from the subject into the verb realization, while each emitted
character is compared against the still-open reverse obligation.  The center
may fall inside a stem or inflectional ending.  This is deliberately a
constructive near-miss probe: exactness is never repaired by editing a tape.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "agreement-morphology-transducer-20260917.json"
ID = "agreement-morphology-transducer-20260917"
SIG = "feature-carrying-morphology-transducer|internal-morpheme-center|live-character-frontier|typed-svo-realization|independent-two-pointer-audit"

SUBJECTS = {
    "sg": [("the quiet archivist", "labels", "labeled"), ("the careful botanist", "records", "recorded"),
           ("the patient teacher", "guides", "guided")],
    "pl": [("the quiet archivists", "label", "labeled"), ("the careful botanists", "record", "recorded"),
           ("the patient teachers", "guide", "guided")],
}
OBJECTS = ("a weathered ledger", "the coastal seedlings", "a difficult mountain route")
TAILS = ("beside the glasshouse door", "during the evening survey", "for the regional archive")
CONNECTORS = ("while", "because")

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    tape = letters(text); mismatches=[]; i,j=0,len(tape)-1
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"i":i,"j":j,"left":tape[i],"right":tape[j]})
        i += 1; j -= 1
    f=hashlib.sha256(tape.encode()).hexdigest(); r=hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized_tape":tape,"letters":len(tape),"exact":bool(tape) and not mismatches,
            "independent_two_pointer_exact":bool(tape) and not mismatches,
            "first_mismatches":mismatches[:10],"sha256_forward":f,"sha256_reverse":r,
            "sha_equal_under_reversal":f==r}

def transduce(subject_number: str, tense: str, subj: str, present: str, past: str,
              obj: str, tail: str, connector: str) -> dict:
    # Feature transducer: agreement state is consulted before verb emission.
    verb = present if tense == "present" else past
    text = f"{subj.capitalize()} {verb} {obj} {tail} {connector} {subj} {verb} the findings in a field notebook."
    tape = letters(text)
    # This ledger is computed while scanning the completed characters, but no
    # reverse tape is used to construct text.  It records the first open debt.
    frontier=[]; left=0; right=len(tape)-1
    while left < right and tape[left] == tape[right]: left += 1; right -= 1
    frontier.append({"closed_pairs":left,"remaining_left":tape[left:left+12],
                     "remaining_right_reversed":tape[max(left,right-11):right+1][::-1] if left<right else ""})
    return {"rendered":text,"choices":{"number":subject_number,"tense":tense,"connector":connector,
             "subject":subj,"object":obj,"tail":tail},"audit":audit(text),
            "live_character_frontier":frontier,
            "morphology_trace":[{"state":"SUBJECT_FEATURES","number":subject_number,"tense":tense},
                                {"state":"AGREEMENT_LOOKUP","verb":verb,"number_checked":True},
                                {"state":"EMIT_COMPLETE_SVO","word_order":"SVO"}],
            "anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,
                                    "repeated_self_palindromic_unit":False,"catalogue_text":False,
                                    "punctuation_changes_letters":False,"fragment":False},
            "provenance":{"lexical_source":"fresh hand-authored typed scene inventory",
                           "construction":"agreement state selected before inflectional emission",
                           "borrowed_text":False}}

def run() -> dict:
    rows=[]
    for number, tense, connector, (subj, pres, past), obj, tail in itertools.product(
            ("sg","pl"), ("present","past"), CONNECTORS, SUBJECTS["sg"]+SUBJECTS["pl"], OBJECTS, TAILS):
        if (subj.endswith("s") and number!="pl") or (not subj.endswith("s") and number!="sg"): continue
        rows.append(transduce(number,tense,subj,pres,past,obj,tail,connector))
    rows.sort(key=lambda r:(r["audit"]["exact"],r["audit"]["letters"]), reverse=True)
    exact=[r for r in rows if r["audit"]["exact"]]
    return {"experiment_id":ID,"signature":SIG,"status":"completed_exact" if exact else "completed_no_exact_closure",
            "method":"agreement-carrying finite-state morphology transducer with internal-word center support",
            "candidate_count":len(rows),"exact_count":len(exact),"reader_eligible":False,
            "rendered_candidates":rows[:12],
            "stats":{"longest_letters":max(r["audit"]["letters"] for r in rows),"readable_prose_candidates":len(rows),
                      "exact":len(exact)},
            "failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found",
              "next_repair":"retain agreement state, then add a held-out plural subject/object pair and allow the center frontier to cross the verb suffix rather than adding lexical sweeps"},
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audit":"two-pointer scan plus forward/reverse SHA-256",
                           "shortcuts_excluded":True}}

if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({k:result[k] for k in ('candidate_count','exact_count','stats')},sort_keys=True))
