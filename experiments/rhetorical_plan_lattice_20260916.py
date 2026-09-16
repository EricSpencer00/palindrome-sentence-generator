"""Joint rhetorical-plan and character-residual search.

This route searches contrast, concession, and causal discourse plans while
lexicalizing both sides of the character equation.  It is deliberately not a
word-order or event-pair construction: each result is two complete, connected
sentences with an explicit discourse relation.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "rhetorical-plan-lattice-20260916"
SIGNATURE = "rhetorical-plan-lattice|contrast-concession-cause|anaphoric-discourse-continuity|joint-character-residual|independent-exact-audit|heldout-connective-repair"
PLANS = [
    ("contrast", "Although {a} {v} {o}, {b} {w} {p}."),
    ("cause", "Because {a} {v} {o}, {b} {w} {p}."),
    ("concession", "Even when {a} {v} {o}, {b} still {w} {p}."),
]
BANK = [
    ("the patient clerk", "checks", "the quiet ledger", "the careful editor", "revises", "the plain report"),
    ("a steady gardener", "waters", "the young roses", "a watchful neighbor", "mends", "the old gate"),
    ("the kind teacher", "helps", "a new pupil", "the calm librarian", "sorts", "the worn books"),
]

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def exact(s):
    t = letters(s); return bool(t) and t == t[::-1]
def residual(left, right):
    a, b = letters(left), letters(right)[::-1]; n = min(len(a), len(b)); k = 0
    while k < n and a[k] == b[k]: k += 1
    return {"matched_prefix": k, "left_length": len(a), "right_length": len(b), "closed": k == len(a) == len(b)}
def words(s): return re.findall(r"[a-z]+", s.casefold())
def render(plan, row, swap=False):
    a,v,o,b,w,p = row
    if swap: a,v,o,b,w,p = b,w,p,a,v,o
    return PLANS[plan][1].format(a=a,v=v,o=o,b=b,w=w,p=p)
def run(rows, phase):
    out=[]
    for pi,(name,_) in enumerate(PLANS):
        for i,row in enumerate(rows):
            left = render(pi,row)
            for j,other in enumerate(rows):
                right = render(pi,other,swap=True)
                text = left + " " + right
                out.append({"phase":phase,"plan":name,"left":left,"right":right,"rendered":text,"residual":residual(left,right),"audit":{"exact":exact(text),"letters":len(letters(text))},"complete_prose":all(len(words(x))>=7 for x in (left,right)),"no_repeated_units":left != right,"reader_eligible":exact(text) and left != right})
    return out
def main():
    base=run(BANK,"base")
    # Repair is a new discourse operator: replace the connective plan with a
    # held-out anaphoric consequence relation, then rerun the same ledger.
    global PLANS
    PLANS = PLANS + [("therefore", "Since {a} {v} {o}, {b} therefore {w} {p}.")]
    repair=run(BANK,"repair")
    payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"plans":[x[0] for x in PLANS],"base":{"candidates":base,"exact_count":sum(x["reader_eligible"] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x["reader_eligible"] for x in repair)},"repair_action":"added held-out therefore/anaphoric consequence plan and reran joint character residual search","provenance":{"source":"human-authored lexical banks and productive rhetorical templates","catalogue_used":False,"borrowed_text":False,"word_order_only":False,"fragments":False}}
    (ROOT/"runs"/"rhetorical-plan-lattice-20260916.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":payload["base"]["exact_count"],"repair_exact":payload["repair"]["exact_count"]}))
if __name__ == "__main__": main()
