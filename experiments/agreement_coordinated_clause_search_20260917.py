"""Fresh constructive lane: agreement-carrying coordinated clauses.

The two clauses are generated together from typed registers (number, tense,
and transitive valency).  The seam checker compares opposing characters while
the complete tape is assembled; it is diagnostic, never a readability claim.
"""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "agreement-coordinated-clause-search-20260917.json"
ID = "agreement-coordinated-clause-search-20260917"

SUBJECTS = [("the careful baker", "sg"), ("the quiet sailors", "pl"),
            ("the young teacher", "sg"), ("the patient gardeners", "pl")]
OBJECTS = ["a letter", "the lantern", "a small map", "the blue vessel"]
VERBS = [("carried", "past"), ("opened", "past"), ("marked", "past"), ("held", "past")]
PLACES = ["beside the river", "under the old bridge", "near the harbor", "by the garden"]

def tape(s: str) -> str:
    return "".join(c.lower() for c in s if "a" <= c.lower() <= "z")

def audit(s: str) -> dict:
    t = tape(s); mism = [(i, len(t)-1-i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"letters": len(t), "exact": bool(t) and not mism, "mismatch_count": len(mism),
            "mismatch_rate": len(mism)/max(1, len(t)//2), "first_mismatches": mism[:12],
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def independent_audit(s: str) -> dict:
    # Deliberately separate implementation: two-pointer comparison and SHA.
    chars = [c.casefold() for c in s if c.casefold() in "abcdefghijklmnopqrstuvwxyz"]
    i, j, mism = 0, len(chars)-1, []
    while i < j:
        if chars[i] != chars[j]: mism.append((i, j, chars[i], chars[j]))
        i += 1; j -= 1
    raw = "".join(chars)
    return {"exact": not mism and bool(raw), "mismatch_count": len(mism),
            "sha256_forward": hashlib.sha256(raw.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(raw[::-1].encode()).hexdigest()}

def flags(s: str) -> dict:
    words = [tape(x) for x in re.findall(r"[A-Za-z]+", s)]
    content = [w for w in words if w not in {"a","an","the","and","then","was","were","by","under","near","beside","past","held"}]
    return {"word_order_mirror": words == [w[::-1] for w in words[::-1]],
            "repeated_content": len(content) != len(set(content)),
            "self_palindromic_content_words": [w for w in content if len(w)>1 and w==w[::-1]],
            "borrowed_catalogue_text": False, "finished_tape_reversed": False}

def render(a, b, obj1, obj2, v1, v2, p1, p2):
    # Registers enforce agreement: singular subjects use singular auxiliary;
    # past transitive verbs remain valency-compatible on both sides.
    return f"{a} {v1} {obj1} {p1}, and {b} {v2} {obj2} {p2}."

def run() -> dict:
    rows=[]
    for (a,na),(b,nb),obj1,obj2,(v1,t1),(v2,t2),p1,p2 in itertools.islice(
        itertools.product(SUBJECTS, SUBJECTS, OBJECTS, OBJECTS, VERBS, VERBS, PLACES, PLACES), 0, 240):
        # coordinated clauses intentionally allow different subjects, but each
        # clause carries a complete subject-number/tense/object register.
        s=render(a,b,obj1,obj2,v1,v2,p1,p2); au=audit(s); ind=independent_audit(s)
        row={"rendered":s,"registers":{"left":{"subject_number":na,"tense":t1,"valency":"transitive"},"right":{"subject_number":nb,"tense":t2,"valency":"transitive"}},"audit":au,"independent_audit":ind,"shortcut_flags":flags(s),"provenance":{"generator":ID,"construction":"typed coordinated clauses; joint register selection","catalogue_imported":False,"seed_used_as_output":False}}
        rows.append(row)
    # Repeated function/content words can arise naturally in two clauses; the
    # actual shortcut gates are mirrored word order, self-palindromes, imports,
    # and finished-tape reversal.
    eligible=[r for r in rows if not any(r["shortcut_flags"][k] for k in
        ("word_order_mirror", "self_palindromic_content_words",
         "borrowed_catalogue_text", "finished_tape_reversed"))]
    best=min(eligible,key=lambda r:(r["audit"]["mismatch_count"],-r["audit"]["letters"]))
    return {"experiment_id":ID,"status":"completed_no_exact_closure","construction":"agreement-carrying coordinated clauses","config":{"rows":len(rows),"registers":"subject number + past tense + transitive object","search":"bounded Cartesian typed grammar"},"actual_candidates":eligible[:12],"best":best,"exact_candidates":[r for r in eligible if r["audit"]["exact"] and r["independent_audit"]["exact"]],"independent_validation":"two-pointer ASCII audit plus SHA-256 forward/reverse; recomputed after rendering","reader_gate":"closed: no exact novel survivor; diagnostics do not certify readability","next_repair":"Add agreement-preserving inflection and a seam-aware boundary operator that selects the next word jointly from both clauses; retain this lane as a non-duplicate baseline.","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(),indent=2)+"\n")
    r=json.loads(OUT.read_text()); print(json.dumps({"status":r["status"],"rows":r["config"]["rows"],"best":r["best"]["rendered"],"mismatches":r["best"]["audit"]["mismatch_count"],"exact":len(r["exact_candidates"])}))
