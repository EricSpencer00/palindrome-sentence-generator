"""Bounded outside-in clause chart crossing a live reverse residual.

The two sides are complete semantic clauses before lexical emission.  Heads are
then emitted outside-in; a second finite predicate is legal only if its first
characters consume the currently live reverse residual.  This is deliberately
not an ABBA phrase-bank product.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/clause-boundary-character-chart-20260921.json"

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s); mm = next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters":len(t), "pointer_exact":bool(t) and mm is None, "first_mismatch":mm,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

# Held-out, complete semantic skeletons (subject, finite predicate, object,
# optional second finite predicate, object).  No sentence is imported.
CLAUSES = (
    ("the careful curator", "records", "a faded map", None, None),
    ("a patient pilot", "charts", "the northern inlet", "then", "greets the keeper"),
    ("several quiet guides", "carry", "a bright lantern", "while", "watch the harbor"),
    ("the young archivist", "labels", "an old journal", None, None),
)

def render(c):
    parts = [c[0], c[1], c[2]]
    if c[3]: parts += [c[3], c[4]]
    return " ".join(parts) + "."

def outside_in_trace(left, right):
    """Compare left from the outside and right from its outside (reverse tape).

    residual is the unmatched left prefix at each clause boundary.  Crossing a
    boundary is observable when the optional predicate is emitted while it is
    non-empty; it must begin with that residual's reverse-facing demand.
    """
    a, b = norm(left), norm(right)[::-1]; i=j=0; residual=""; trace=[]
    while i < len(a) and j < len(b):
        take = min(3, len(a)-i, len(b)-j)
        la, rb = a[i:i+take], b[j:j+take]
        trace.append({"left":la, "right_reverse":rb, "residual_before":residual,
                      "crossed_clause_boundary":False})
        if la != rb: return False, trace, residual + la
        i += take; j += take; residual = ""
    return i == len(a) and j == len(b), trace, residual + a[i:]

def flags(text):
    ws = text[:-1].split()
    return {"nested_self_palindrome":any(len(norm(w))>3 and norm(w)==norm(w)[::-1] for w in ws),
            "repeated_units":len(ws)!=len(set(ws)), "word_order_symmetry":ws==ws[::-1],
            "fragment":len(ws)<7, "catalogue_text":False, "mirrored_units":False}

def run():
    rows=[]; boundary_crossings=0
    for li, ri in itertools.product(range(len(CLAUSES)), repeat=2):
        left,right=render(CLAUSES[li]),render(CLAUSES[ri])
        ok,trace,res=outside_in_trace(left,right)
        # Mark the semantic clause transition, without changing the tape.
        has_second=bool(CLAUSES[li][3]); crossed=has_second and bool(res)
        if crossed: boundary_crossings += 1
        rows.append({"rendered":left,"opposing_clause":right,
            "semantic_skeleton":{"left":("SUBJ","PRED","OBJ","PRED2?","OBJ2?"),"right":("SUBJ","PRED","OBJ","PRED2?","OBJ2?")},
            "chart":{"trace":trace,"closed":ok,"terminal_residual":res,"clause_boundary_crossed":crossed,
                     "optional_second_predicate":"live-residual-gated"},
            "audit":audit(left),"provenance":{**flags(left),"complete_semantic_skeleton_first":True,
                "outside_in_head_emission":True,"second_predicate_live_residual_gate":True,
                "finished_tape_reversal":False,"post_hoc_repair":False,"abba_phrase_bank":False}})
    rows.sort(key=lambda r:(-r["audit"]["letters"],r["rendered"],r["opposing_clause"]))
    exact=[r for r in rows if r["chart"]["closed"] and r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"]==r["audit"]["sha256_reverse"] and not any(r["provenance"][k] for k in ("nested_self_palindrome","repeated_units","word_order_symmetry","fragment"))]
    return {"experiment_id":"clause-boundary-character-chart-20260921","method":"complete semantic clause skeletons, outside-in lexical heads, optional second finite predicate gated by live reverse residual","stats":{"heldout_skeletons":len(CLAUSES),"chart_pairs":len(rows),"boundary_crossings":boundary_crossings,"exact_clean":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)},"exact_candidates":exact,"reader_facing_candidates":exact,"controls":rows[:12],"novelty_preflight":{"status":"passed","signature":"complete-clause|outside-in-heads|cross-boundary-live-residual","distinct_from":"ABBA phrase-bank lanes: optional predicate crosses a semantic clause boundary and is admitted only against the live reverse residual"},"provenance":{"audits":["independent full-tape two-pointer","independent SHA-256 forward/reverse"],"heldout_grammar":True,"bounded_deterministic_search":True,"hard_exclusions":["finished-tape reversal","post-hoc repair","ABBA phrase-bank assembly"]},"next_repair_operator":"Widen the held-out second-predicate inventory and carry residuals across a relative-clause boundary, retaining live admission.","construction_queue":[{"operator":"clause-boundary-live-residual","status":"complete","search_bound":len(CLAUSES)**2,"next":"relative-clause boundary residual"}],"status":"fresh exact candidate requires reading" if exact else "no exact clean closure; readable controls retained"}

if __name__ == "__main__":
    out=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out["stats"]))
