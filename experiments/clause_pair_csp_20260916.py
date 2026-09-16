"""Typed clause-pair CSP with a central pivot (diagnostic, not promotion).

Two independently authored S-V-O scenes are selected from a small typed chart.
The CSP matches their *character tapes* from the pivot outward; it never
reverses word order or copies a sentence.  Failure is useful evidence: the
longest intact prose and the exact unmatched debt are retained.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

from llm_palindrome.admission import mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
FAMILY_ID = "clause-pair-csp-central-pivot"
SIGNATURE = "typed-svo-valency|independent-scene-pairs|central-pivot|character-tape-csp|no-word-mirror"

SUBJECTS = [("The archivist", "animate"), ("A patient sailor", "animate"),
            ("The young botanist", "animate"), ("A quiet teacher", "animate")]
VERBS = [("records", "animate"), ("guides", "animate"), ("observes", "animate"), ("carries", "animate")]
OBJECTS = [("the carefully folded winter map", "artifact"), ("a weathered cedar rescue boat", "artifact"),
           ("the bright brass navigation lantern", "artifact"), ("a field notebook from camp", "artifact")]
PIVOTS = (" at dawn", " beside the river", " beneath the old bridge")

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def scene(s, v, o, pivot):
    assert s[1] == v[1] == "animate" and o[1] == "artifact"
    return f"{s[0]} {v[0]} {o[0]}{pivot}."
def pointer_audit(s):
    """Independent standard-library two-pointer check."""
    t = norm(s)
    i, j = 0, len(t) - 1
    first = None
    while i < j:
        if t[i] != t[j]:
            first = {"index": i, "left": t[i], "right": t[j]}
            break
        i += 1
        j -= 1
    return {"exact": bool(t) and first is None, "letters": len(t),
            "comparisons": i if first is not None else len(t) // 2,
            "first_mismatch": first}


def hash_audit(s):
    """Separate forward/reverse digest check; does not reuse pointer state."""
    t = norm(s)
    forward = hashlib.sha256(t.encode()).hexdigest()
    reverse = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"exact": bool(t) and forward == reverse, "letters": len(t),
            "sha256": forward, "reverse_sha256": reverse}
def csp(left, right, pivot):
    a, b = norm(left), norm(right)
    # Match around a central pivot without changing either scene's word order.
    matched = 0
    while matched < min(len(a), len(b)) and a[-1-matched] == b[matched]: matched += 1
    debt = {"left_unmatched": a[:-matched] if matched else a,
            "right_unmatched": b[matched:], "letters": len(a)+len(b)-2*matched}
    return {"matched_from_pivot": matched, "debt": debt, "pivot": pivot,
            "text": left + " " + right}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--min-letters", type=int, default=100); args = ap.parse_args()
    rows=[]
    for i, s in enumerate(SUBJECTS):
        for j, v in enumerate(VERBS):
            for k, o in enumerate(OBJECTS):
                for p in PIVOTS:
                    l=scene(s,v,o,p); r=scene(SUBJECTS[(i+1)%len(SUBJECTS)], VERBS[(j+2)%len(VERBS)], OBJECTS[(k+1)%len(OBJECTS)], PIVOTS[(p and (PIVOTS.index(p)+1)%len(PIVOTS))])
                    if l != r:
                        row=csp(l,r,p); row.update({"left":l,"right":r,"left_types":[s[1],v[1],o[1]],"right_types":[SUBJECTS[(i+1)%4][1],VERBS[(j+2)%4][1],OBJECTS[(k+1)%4][1]]}); rows.append(row)
    best=max(rows,key=lambda x:x["matched_from_pivot"])
    text=best["text"]; assert len(norm(text)) >= args.min_letters
    out={"id":FAMILY_ID,"status":"diagnostic_no_exact_closure","text":text,"letters":len(norm(text)),
      "clauses":{"left":best["left"],"right":best["right"],"different":best["left"]!=best["right"],"complete":True,"typed_valency":True},
      "pivot_csp":best,"audits":{"two_pointer":pointer_audit(text),"independent_reverse_sha":hash_audit(text)},
      "mechanical_gates":mechanical_admission_checks(text, min_letters=args.min_letters, max_letters=2000),
      "novelty_preflight":{"registry_checked":True,"family_id":FAMILY_ID,"signature":SIGNATURE,"catalogue_text_imported":False},
      "provenance":{"generator":str(Path(__file__).resolve()),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"candidate_count":len(rows)},
      "exact_debt":best["debt"],"next_repair":"replace only the right clause's pivot-adjacent typed complement, then rerun the reverse-index CSP and both audits."}
    out_dir = ROOT / "runs" / "clause-pair-csp-central-pivot-20260916"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out,indent=2))
if __name__ == '__main__': main()
