"""Semantic-center SAT: fix an authored event, then solve bilateral role slots.

This deliberately does not reverse a finished sentence or reuse a palindrome.
The SAT variables are lexical choices for subject/verb/object slots; clauses are
character obligations against the opposite side.  The current inventory is
small, so the artifact is also useful as a transparent failure/repair record.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "semantic-center-sat-20260916"
SIGNATURE = "authored-nonpalindromic-center-event|bilateral-role-slot-sat|global-character-equation|semantic-valency-gate|heldout-center-repair|independent-exact-hash-audit"
OUT = ROOT / "runs" / (EXPERIMENT_ID + ".json")

CENTERS = [
    {"id":"market", "text":"at dusk the nurse closed the market ledger", "sense":"a nurse closes a ledger at dusk", "roles": ("the nurse", "closed", "the ledger")},
    {"id":"harbor", "text":"at dawn the pilot marked the harbor chart", "sense":"a pilot marks a chart at dawn", "roles": ("the pilot", "marked", "the chart")},
]
SUBJECTS = ("the careful nurse", "the patient pilot", "the quiet keeper", "the young farmer")
VERBS = ("carried", "opened", "marked", "checked")
OBJECTS = ("a sealed letter", "the old ledger", "the harbor chart", "a warm parcel")

def exact_audit(text: str) -> dict:
    letters = normalize_letters(text)
    reverse = letters[::-1]
    return {"letters": len(letters), "forward": letters, "reverse": reverse,
            "exact": letters == reverse,
            "hash_forward": hashlib.sha256(letters.encode()).hexdigest(),
            "hash_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
            "hash_equal": hashlib.sha256(letters.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest(),
            "independent_pointer_audit": all(letters[i] == letters[-1-i] for i in range(len(letters))) }

def sat_score(left: str, right: str) -> dict:
    """Count satisfied global character clauses without claiming exactness."""
    a, b = normalize_letters(left), normalize_letters(right)[::-1]
    n = min(len(a), len(b)); matches = sum(a[i] == b[i] for i in range(n))
    return {"paired_positions": n, "satisfied_clauses": matches,
            "total_clauses": max(len(a), len(b)), "residual_debt": abs(len(a)-len(b)) + n-matches}

def realize(center: dict, subject: str, verb: str, obj: str, label: str) -> dict:
    # The center is fixed before slot assignment; the two role frames are
    # independently authored, not a reverse-tape rendering.
    left = f"{subject} {verb} {obj}"
    right = center["text"]
    rendered = left + ". " + right + "."
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=220)
    audit = exact_audit(rendered)
    return {"label":label, "rendered":rendered, "letters":audit["letters"],
            "center_event":center["text"], "center_sense":center["sense"],
            "slot_assignment":{"subject":subject,"verb":verb,"object":obj},
            "sat_character_equation":sat_score(left, right), "exact_audit":audit,
            "checks":checks, "admitted":bool(audit["exact"] and all(checks.values())),
            "provenance":{"center_authored":True,"source_sentences_copied":False,
                           "catalogue_imported":False,"reversed_finished_sentence":False,
                           "repeated_self_palindromic_unit":False,"word_order_symmetry":False}}

def novelty():
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    atoms=set(re.findall(r"[a-z0-9]+",SIGNATURE)); rows=[]
    for x in reg["entries"]:
        shared=sorted(atoms & set(re.findall(r"[a-z0-9]+",x["signature"])))
        rows.append({"id":x["id"],"shared_atoms":shared})
    return {"exact_signature_collision":any(x["signature"]==SIGNATURE for x in reg["entries"]),
            "registry_entries":len(reg["entries"]),"nearest":sorted(rows,key=lambda x:(-len(x["shared_atoms"]),x["id"]))[:5],
            "preflight_rule":"reject exact collision or a center-out/scene-stack replay"}

def run():
    center=CENTERS[0]
    probes=[realize(center,s,v,o,"center-fixed-slot-probe") for s,v,o in itertools.islice(itertools.product(SUBJECTS,VERBS,OBJECTS),8)]
    repair=realize(CENTERS[1],"the observant pilot","checked","the harbor chart","held-out-center-repair")
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed",
            "method":"fix a non-palindromic authored event first; assign independent semantic role slots; score global character clauses; reject on exact pointer/hash audit",
            "novelty_preflight":novelty(),"candidates":probes+[repair],
            "stats":{"candidates":len(probes)+1,"admitted":sum(int(x["admitted"]) for x in probes+[repair]),"exact":sum(int(x["exact_audit"]["exact"]) for x in probes+[repair])},
            "next_repair":"expand held-out verb/object paradigms while preserving the fixed event and add a typed seam CSP for residual character debt",
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}

if __name__ == "__main__":
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    payload=run(); OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))
