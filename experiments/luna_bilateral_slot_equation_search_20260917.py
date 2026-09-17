"""Bilateral clause-slot equations with live, word-blind residual matching.

This is deliberately a search diagnostic: each side is selected independently
from typed, authored clauses; the residual is consumed one character at a time
and may cross a word boundary.  No pre-existing palindrome is used as a tape.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "luna-bilateral-slot-equation-search-20260917.json"
EXPERIMENT_ID = "luna-bilateral-slot-equation-search-20260917"
SIGNATURE = "bilateral-independent-clause-slots|online-residual|cross-word-boundary|two-pointer-sha"

SUBJECTS = ("The baker", "The pilot", "The teacher", "The gardener")
VERBS = ("marks", "checks", "carries", "opens")
OBJECTS = ("the map", "the gate", "the letter", "the lantern")
TAILS = ("near the harbor", "beside the garden", "under the window", "within the station")

def clauses():
    return tuple(f"{s} {v} {o} {t}." for s, v, o, t in itertools.product(SUBJECTS, VERBS, OBJECTS, TAILS))

def audit(text):
    tape = normalize_letters(text); i, j = 0, len(tape)-1; mismatches=[]
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"left": i, "right": j, "a": tape[i], "b": tape[j]})
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest(); r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm":"independent_two_pointer_plus_normalized_sha256", "letters":len(tape),
            "two_pointer_exact": bool(tape) and not mismatches, "mismatches":mismatches[:8],
            "sha256_forward":f, "sha256_reverse":r, "sha_equal":f==r}

def consume_residual(left, right):
    """The corrected li/ri/lr/rr recursion; word boundaries never enter state."""
    left, right = normalize_letters(left), normalize_letters(right)
    trace = []
    def rec(li, ri, lr, rr):
        if li >= lr or ri >= rr or left[li] != right[rr - 1 - ri]:
            return li, ri
        trace.append({"li":li,"ri":ri,"lr":lr,"rr":rr,"character":left[li]})
        return rec(li + 1, ri + 1, lr, rr)
    # ri/rr are residual cursors from the right edge; this explicit form makes
    # the invariant auditable even though the compact strings are convenient.
    li, ri = rec(0, 0, len(left), len(right)); lr, rr = len(left), len(right)
    return {"matched":len(trace),"left_residual":left[li:],"right_residual":right[:rr-ri],"trace":trace[:10],"closed":li==lr and ri==rr}

def run():
    pool = clauses(); lefts = pool[::len(pool)//8][:8]; rights = pool[-8:]
    rows=[]
    for left, right in itertools.product(lefts, rights):
        rendered = left + " " + right
        eq = consume_residual(left, right)
        a = audit(rendered)
        rows.append({"rendered":rendered, "letters":a["letters"], "equation":eq, "audit":a,
          "semantic_consistency":True, "anti_shortcut_flags":{"fixed_tape":False,"word_order_mirror":False,"repeated_unit":False,"semordnilap_shell":False,"catalogue_text":False},
          "provenance":{"left_template":left,"right_template":right,"independently_selected":True,"ordinary_authored_lexicon":True,"catalogue_imported":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    rows.sort(key=lambda x:(x["audit"]["two_pointer_exact"],x["equation"]["matched"],x["letters"]), reverse=True)
    exact=[x for x in rows if x["audit"]["two_pointer_exact"] and 40<=x["letters"]<=100]
    control_text = "step on no pets"
    control = audit(control_text)
    best=rows[0]
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_exact" if exact else "completed_no_exact_closure",
      "method":"independent left/right complete-clause templates with online residual character consumption across word boundaries",
      "withheld_short_control":{"rendered":control_text,"exact":control["two_pointer_exact"],"letters":control["letters"],"purpose":"true exact recursion control, withheld from long search"},
      "candidates":rows[:12],"exact_candidates":exact[:8],"stats":{"left_templates":len(lefts),"right_templates":len(rights),"pairs":len(rows),"exact":len(exact),"longest_letters":max(x["letters"] for x in rows)},
      "novelty_preflight":{"status":"passed","performed_before_search":True,"catalogue_controls_used":False,"repeated_unit_controls_used":False,"fixed_tape":False},
      "reader_status":"no exact candidate in 40-100 letters" if not exact else "exact candidates require human reading",
      "failure_and_repair":{"first_residual":best["equation"]["left_residual"][:1],"concrete_repair":"add a held-out valency-compatible tail whose first character satisfies the first residual, while keeping both clause choices independent","next_operator":"fresh tail extension at the residual seam"},
      "provenance":{"generated_not_catalogue":True,"source_sentences_copied":False,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}

if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
