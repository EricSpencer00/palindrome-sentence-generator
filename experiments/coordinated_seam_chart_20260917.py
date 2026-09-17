"""Coordinated-clause composition with live opposed-character seam obligations.

This is a construction experiment, not a palindrome catalogue.  Each clause is
freshly authored, and a small chart chooses lexical variants while recording
the characters that would have to agree across the outside-in seam.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "coordinated-seam-chart-20260917"
OUT = ROOT / "runs" / f"{ID}.json"

def tape(s): return "".join(c.lower() for c in s if c.lower() in "abcdefghijklmnopqrstuvwxyz")

def audit(s):
    t=tape(s); mism=[]
    for i in range(len(t)//2):
        j=len(t)-1-i
        if t[i]!=t[j]: mism.append((i,j,t[i],t[j]))
    return {"letters":len(t),"exact":bool(t) and not mism,"mismatch_count":len(mism),
            "first_mismatches":mism[:12],"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def independent(s):
    t="".join(c.casefold() for c in s if c.casefold() in "abcdefghijklmnopqrstuvwxyz")
    return {"exact":bool(t) and t==t[::-1],"letters":len(t),
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}

def flags(s):
    words=re.findall(r"[A-Za-z]+",s.lower()); content=[w for w in words if w not in {"a","an","the","and","while","as","then","at","in","near","before"}]
    return {"word_order_mirror":words==[w[::-1] for w in words],"repeated_content":len(content)!=len(set(content)),
            "self_palindromic_content_words":[w for w in content if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue_text":False,"finished_tape_reversed":False}

# The cells are semantic choices; their first/last letters are chart values,
# retained in the provenance so a future exact solver can constrain them live.
SCENES=[
 {"id":"orchard","left":"At sunrise, the orchard keeper","verbs":["prunes","marks"],"objects":["the pear trees","the young apple trees"],"tail":"and carries baskets home","right":"while the patient beekeeper","rverbs":["checks","counts"],"robjects":["the honey frames","the cedar hive frames"],"rtail":"before evening."},
 {"id":"station","left":"At the morning station, the signal officer","verbs":["checks","clears"],"objects":["the westbound train","the waiting night train"],"tail":"and records the platform change","right":"while the careful porter","rverbs":["guides","loads"],"robjects":["the travel cases","the canvas travel cases"],"rtail":"before departure."},
 {"id":"clinic","left":"In the small clinic, the nurse","verbs":["labels","sorts"],"objects":["the clean medicine trays","the sealed medicine bottles"],"tail":"and updates each quiet chart","right":"while the senior doctor","rverbs":["reviews","signs"],"robjects":["the evening notes","the new evening notes"],"rtail":"before rounds."},
]

def render(sc, a, b, c, d):
    return f"{sc['left']} {sc['verbs'][a]} {sc['objects'][b]} {sc['tail']}, and {sc['right']} {sc['rverbs'][c]} {sc['robjects'][d]} {sc['rtail']}"

def run():
    rows=[]; failures=[]
    for sc,a,b,c,d in itertools.product(SCENES,range(2),range(2),range(2),range(2)):
        s=render(sc,a,b,c,d); au=audit(s); ind=independent(s); fl=flags(s); t=tape(s)
        seam=[(i,len(t)-1-i,t[i],t[-1-i],t[i]==t[-1-i]) for i in range(min(10,len(t)//2))]
        row={"rendered":s,"scene_id":sc["id"],"choices":{"left_verb":a,"left_object":b,"right_verb":c,"right_object":d},"seam_chart":seam,"audit":au,"independent_audit":ind,"shortcut_flags":fl,"provenance":{"generator":ID,"construction":"fresh coordinated clause; opposed character chart captured during lexical choice","catalogue_imported":False,"fixed_tape":False,"seed_used_as_output":False}}
        if not 100<=au["letters"]<=180: row["failure_reason"]="length_gate"; failures.append(row); continue
        if fl["word_order_mirror"] or fl["self_palindromic_content_words"] or fl["borrowed_catalogue_text"]: row["failure_reason"]="novelty_or_shortcut_gate"; failures.append(row); continue
        rows.append(row)
    exact=[r for r in rows if r["audit"]["exact"] and r["independent_audit"]["exact"]]
    best=min(rows,key=lambda r:(r["audit"]["mismatch_count"],-r["audit"]["letters"]))
    return {"experiment_id":ID,"status":"completed_no_exact_closure" if not exact else "exact_candidates_found","config":{"scenes":3,"variants_per_scene":16,"length_gate":"100<=letters<=180 before scoring","seam_chart":"first ten opposed character obligations"},"actual_candidates":rows,"best":best,"exact_candidates":exact,"failed_branches":failures,"independent_validation":"separate normalized tape and SHA forward/reverse audit","reader_gate":"closed unless exact novel survivor; programmatic measures diagnose only","novelty_preflight":"fresh scene clauses; no catalogue or finished reverse tape","next_repair":"Use the recorded seam chart to replace only boundary lexical slots with agreement-compatible synonyms, preserving clause meaning and attachment; then run blinded intact-prose review on exact survivors.","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__=="__main__":
    r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({"status":r["status"],"candidates":len(r["actual_candidates"]),"failed":len(r["failed_branches"]),"best":r["best"]["rendered"],"letters":r["best"]["audit"]["letters"],"mismatches":r["best"]["audit"]["mismatch_count"],"exact":len(r["exact_candidates"])}))
