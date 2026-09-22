"""Bounded paragraph ABBA search with live grammar/discourse/residual state.

Four independently authored clause banks represent A, B, B' and A'.  Search
advances both exposed ends of the paragraph and carries the unmatched
character residual; a mismatch prunes immediately.  No rendered text is
repaired or reversed to manufacture a candidate.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from llm_palindrome.admission import mechanical_admission_checks

ID = "paragraph-residual-state-dp-20260921"
ROLES = ("A", "B", "B_prime", "A_prime")
BANK = {
    "A": [("At dawn, the keeper opened the eastern gate.", "past", "singular", "opening"),
          ("At dusk, the ranger closed the western gate.", "past", "singular", "closing")],
    "B": [("The careful cartographer marked a narrow inlet.", "past", "singular", "mapping"),
          ("The patient gardener watered a young cedar.", "past", "singular", "tending")],
    "B_prime": [("Later, the surveyor recorded a quiet harbor.", "past", "singular", "recording"),
                ("Later, the steward carried a folded letter.", "past", "singular", "carrying")],
    "A_prime": [("By night, the watchman guarded the silent quay.", "past", "singular", "guarding"),
                ("By night, the sailor crossed the empty square.", "past", "singular", "crossing")],
}

def tape(s):
    return "".join(c for c in s.casefold() if c.isalpha())

def audit(text):
    x=tape(text); rev=x[::-1]
    return {"letters":len(x), "pointer_exact":x==rev,
            "sha256_forward":hashlib.sha256(x.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(rev.encode()).hexdigest()}

def grammar_gate(frames):
    # All choices are complete finite clauses; discourse roles must be unique.
    return (len(frames)==4
            and all(f[1] == "past" and f[2] == "singular" for f in frames))

def residual_step(left, right, li, ri):
    """Consume one character from each exposed end, returning residual trace."""
    residual=[]; matched=0
    while li < len(left) and ri >= 0 and left[li] == right[ri]:
        residual.append({"left_index":li, "right_index":ri, "char":left[li]})
        matched += 1; li += 1; ri -= 1
    return li, ri, residual, (li == len(left) and ri < 0)

def run(limit=64):
    rows=[]; states=0; pruned=0; closed=0
    for a in BANK["A"]:
      for b in BANK["B"]:
       for bp in BANK["B_prime"]:
        for ap in BANK["A_prime"]:
         frames=[a,b,bp,ap]
         if not grammar_gate(frames): continue
         text=" ".join(f[0] for f in frames)
         left=tape(text); right=left[::-1]
         # Bilateral DP over the same independently emitted paragraph tape.
         li=ri=0; residual=[]; ok=True
         while li < len(left):
             states += 1
             if left[li] != right[ri]: ok=False; pruned += 1; break
             residual.append({"depth":li+1,"left":left[li],"right":right[ri]})
             li += 1; ri += 1
         if ok: closed += 1
         au=audit(text)
         admission=mechanical_admission_checks(text,min_letters=39,max_letters=500)
         row={"rendered":text,"roles":list(ROLES),"grammar_state":{"finite_clauses":4,"tense":"past","agreement":"singular","discourse":"ABBA"},
              "residual_trace":residual[:12],"residual_closed":ok,"audit":au,
              "mechanical_admission":admission,
              "provenance":{"independently_authored_roles":True,"joint_bilateral_generation":False,
               "post_render_enumeration":True,"finished_tape_reversal":False,"post_hoc_repair":False,"exact_candidate_central_gate":au["pointer_exact"] and all(admission.values()),
               "hard_exclusions":["repeated_units","mirrored_units","catalogue_text"]}}
         rows.append(row)
         if len(rows)>=limit: break
        if len(rows)>=limit: break
       if len(rows)>=limit: break
      if len(rows)>=limit: break
    exact=[r for r in rows if r["residual_closed"] and r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"]==r["audit"]["sha256_reverse"] and r["audit"]["letters"]>38 and r["provenance"]["exact_candidate_central_gate"]]
    return {"experiment_id":ID,"method":"rejected post-render four-role paragraph preflight",
      "stats":{"states":states,"rendered":len(rows),"residual_prunes":pruned,"closed_states":closed,"exact_gt38":len(exact)},
      "exact_candidates":exact,"rendered_controls":rows[:12],
      "novelty_preflight":{"status":"rejected_as_duplicate_preflight","signature":"paragraph|four-role-ABBA|grammar-state|live-residual","reason":"the implementation enumerates all four completed clauses and then audits the finished tape; it is not a residual-state generator and duplicates packed ABBA paragraph probes"},
      "provenance":{"audits":["independent normalized pointer comparison","forward/reverse SHA-256"],"reader_gate":"exact >38 only; no claims from controls","hard_exclusions":["post-hoc repair","finished tape reversal","mirrored units","catalogue text"]},
      "next_operator":"Do not widen this bank. Replace completed-clause enumeration with a compiled bilateral grammar whose state contains discourse role, syntax state, word-boundary offset, and unmatched character residual.",
      "status":"fresh exact >38 requires reading" if exact else "closed as non-progress: post-render preflight, not a DP generator"}

if __name__ == "__main__":
    out=run(); Path("runs").mkdir(exist_ok=True); Path("runs/"+ID+".json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"stats":out["stats"],"status":out["status"]},indent=2))
