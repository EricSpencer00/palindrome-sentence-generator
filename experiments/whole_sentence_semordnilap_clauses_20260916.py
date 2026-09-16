"""Whole-sentence clause grammar using role-changing semordnilap seams.

Unlike constituent reversal, each arm is independently a complete clause. The
search joins character streams with a live residual; reversible words are
ordinary lexical entries (drawer/reward, part/trap), never copied units.
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FAMILY = "whole-sentence-semordnilap-clauses-20260916"
SIG = "whole-sentence-independent-clause-grammar|semordnilap-lexical-boundaries|role-changing-reversible-words|joint-character-residual-search|independent-complete-prose-audit|no-unit-repetition"

# Subject/verb/object roles are deliberately different on the two sides.
BASE = [
    ("The drawer rewards a poet", "the", "drawer", "rewards", "a poet"),
    ("A part traps the scout", "a", "part", "traps", "the scout"),
    ("The stressed student reads desserts", "the", "stressed student", "reads", "desserts"),
    ("A deliverer meets a reviled man", "a", "deliverer", "meets", "a reviled man"),
]
REPAIR = BASE + [
    ("The drawer parts a reward", "the", "drawer", "parts", "a reward"),
    ("A trap rewards the artist", "a", "trap", "rewards", "the artist"),
]

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def toks(s): return re.findall(r"[a-z]+", s.lower())
def residual_join(left, right):
    """Consume left then right against the reverse-character residual."""
    residual = ""; events=[]
    for side, text in (("left", left), ("right", right)):
        for ch in norm(text):
            if residual:
                event = "cancel" if ch == residual[0] else "contradiction"
                events.append((side, ch, residual[0], event))
                if event == "contradiction": return False, residual, events
                residual = residual[1:]
            else: residual = ch
    return not residual, residual, events
def parse_clause(row):
    ws=toks(row[0]); return len(ws)>=4 and ws[0] in {"a","an","the"} and row[3] in ws and len(set(ws))==len(ws)
def audit(left,right):
    tape=norm(left+" "+right); mism=[]
    for i in range(len(tape)//2):
        if tape[i]!=tape[-1-i]: mism.append({"i":i,"left":tape[i],"right":tape[-1-i]})
    return {"exact":not mism and bool(tape),"letters":len(tape),"mismatch_count":len(mism),"first_mismatches":mism[:5]}
def run(bank, phase):
    rows=[]
    # Every pair is a probe, including failures; no catalogue material or unit reuse.
    for li,l in enumerate(bank):
        for ri,r in enumerate(bank):
            left,right=l[0],r[0]
            ok,res,events=residual_join(left,right)
            a=audit(left,right)
            rows.append({"left":left,"right":right,"left_clause":parse_clause(l),"right_clause":parse_clause(r),"role_boundary":l[2]!=r[2] or l[3]!=r[3],"no_repeated_units":len(set(toks(left+" "+right)))==len(toks(left+" "+right)),"residual_closed":ok,"residual_after":res,"audit":a,"phase":phase,"reader_eligible":ok and parse_clause(l) and parse_clause(r) and a["exact"]})
    return rows
def main():
    base=run(BASE,"base"); repair=run(REPAIR,"repair")
    out={"experiment":FAMILY,"signature":SIG,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"grammar":{"independent_arms":True,"roles":["determiner","subject","finite_verb","object"],"semordnilap_seams":["drawer/reward","part/trap","stressed/desserts","deliverer/reviled"],"joint_character_residual":True,"repeated_units_forbidden":True},"base":{"probes":base,"exact_count":sum(x["reader_eligible"] for x in base)},"repair":{"probes":repair,"exact_count":sum(x["reader_eligible"] for x in repair)},"repair_action":"expanded the clause grammar with part/reward and trap/artist role substitutions, then reran the complete cross-product and independent two-pointer audit","status":"complete"}
    p=ROOT/"runs/whole-sentence-semordnilap-clauses-20260916.json"; p.write_text(json.dumps(out,indent=2)+"\n")
    rp=ROOT/"runs/whole-sentence-semordnilap-clauses-repair-20260916.json"; rp.write_text(json.dumps({"family":FAMILY,"phase":"repair","probes":repair,"exact_count":out["repair"]["exact_count"]},indent=2)+"\n")
    print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":out["base"]["exact_count"],"repair_exact":out["repair"]["exact_count"]}))
if __name__ == "__main__": main()
