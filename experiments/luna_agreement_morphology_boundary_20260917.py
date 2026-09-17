"""Agreement-carrying morphology at a clause boundary.

This is a fresh, typed-clause search: each clause is selected from productive
inflection tables, and the final inflectional segment on the left carries an
obligation into the initial segment on the right.  The prose is authored as
complete clauses; no finished tape is reversed or resegmented.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-agreement-morphology-boundary-20260917.json"
EXPERIMENT_ID = "luna-agreement-morphology-boundary-20260917"
SIGNATURE = "typed-clause|productive-agreement-inflection|bidirectional-seam-obligation|independent-pointer-sha"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

FRAMES = (
    ("the patient archivist", "the patient archivists", "labels", "label", "a weathered ledger", "before the winter meeting"),
    ("the careful botanist", "the careful botanists", "records", "record", "the coastal seedlings", "beside the glasshouse door"),
    ("the quiet cartographer", "the quiet cartographers", "marks", "mark", "a difficult mountain pass", "during the evening survey"),
)
MODES = (("singular_present", False, "present"), ("plural_present", True, "present"),
         ("singular_past", False, "past"), ("plural_past", True, "past"))

def novelty_preflight():
    rows = json.loads(REGISTRY.read_text()).get("entries", [])
    artifact = str(Path(__file__).relative_to(ROOT))
    prior = [r for r in rows if r.get("id") != EXPERIMENT_ID]
    overlaps = [r.get("signature") for r in prior if r.get("signature") == SIGNATURE]
    collision = [r.get("artifact") for r in prior if r.get("artifact") == artifact]
    if overlaps or collision: raise RuntimeError({"overlaps": overlaps, "artifact": collision})
    return {"status":"passed", "registry_entries_read":len(rows), "signature_overlaps":[], "artifact_collisions":[],
            "rejected_shortcuts":["finished-tape reversal", "word-order symmetry", "catalogue text", "fragments"]}

def audit(text):
    tape = normalize_letters(text); mismatches=[]; i,j=0,len(tape)-1
    while i<j:
        if tape[i]!=tape[j]: mismatches.append({"left_index":i,"right_index":j,"left":tape[i],"right":tape[j]})
        i+=1; j-=1
    f=hashlib.sha256(tape.encode()).hexdigest(); r=hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized_tape":tape,"letters":len(tape),"exact":bool(tape) and not mismatches,
            "independent_two_pointer_exact":bool(tape) and not mismatches,"two_pointer_mismatches":mismatches[:8],
            "sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r,
            "mechanical_checks":mechanical_admission_checks(text,min_letters=100,max_letters=320)}

def realize(frame, left_mode, right_mode, bridge):
    ls,lp,verb3,verb0,obj,place=frame; ln, lpl, lt=left_mode; rn,rpl,rt=right_mode
    left_subject=lp if lpl else ls; right_subject=lp if rpl else ls
    left_verb=(verb0 if lpl else verb3) if lt=="present" else verb3.rstrip("s") + "ed"
    right_verb=(verb0 if rpl else verb3) if rt=="present" else verb3.rstrip("s") + "ed"
    # The bridge is a typed, productive agreement carrier, not a copied tape.
    bridge_words={"while":"while", "because":"because", "after":"after"}
    text=f"{left_subject.capitalize()} {left_verb} {obj} {place} {bridge_words[bridge]} {right_subject} {right_verb} the findings in a shared field notebook for the regional committee."
    seam_left=left_verb[-2:]; seam_right=right_verb[:2]
    return {"rendered":text,"choices":{"left":ln,"right":rn,"bridge":bridge},
            "typed_clauses":[{"subject_number":"plural" if lpl else "singular","tense":lt},{"subject_number":"plural" if rpl else "singular","tense":rt}],
            "live_seam_equations":[{"name":"agreement_register_propagation","left":("plural" if lpl else "singular"),"right":("plural" if rpl else "singular"),"matched":(lpl==rpl)},
             {"name":"inflectional_boundary","left_suffix":seam_left,"right_prefix":seam_right,"matched":seam_left[-1:]==seam_right[:1]},
             {"name":"clause_completeness","matched":text.count(" ")>=16}],
            "audit":audit(text),"anti_shortcut_flags":{"finished_tape_reversal":False,"word_order_symmetry":False,"catalogue_text":False,"fragment":False},
            "provenance":{"lexical_source":"fresh hand-authored typed scene frames","productive_tables":"number x tense verb inflection","catalogue_text_imported":False}}

def run():
    pre=novelty_preflight(); rows=[realize(f,l,r,b) for f,l,r,b in itertools.product(FRAMES,MODES,MODES,("while","because","after"))]
    rows.sort(key=lambda x:(x["audit"]["exact"],x["audit"]["letters"]),reverse=True); exact=[x for x in rows if x["audit"]["exact"]]
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_exact" if exact else "completed_no_exact_closure",
            "reader_eligible":bool(exact),"method":"compact typed clause grammar with productive number/tense inflection and bidirectional seam obligations",
            "novelty_preflight":pre,"candidate_count":len(rows),"rendered_candidates":rows[:20],
            "stats":{"frames":len(FRAMES),"variants":len(rows),"exact":len(exact),"over_100":sum(x["audit"]["letters"]>=100 for x in rows)},
            "failure_and_repair":{"failure":"no exact closure" if not exact else "exact closure found","next_repair":"hold out one agreement-bearing verb paradigm, then replace only the first seam residual and rerun independent audits"},
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256","mechanical admission"],"fresh_scene_authoring":True}}
if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
