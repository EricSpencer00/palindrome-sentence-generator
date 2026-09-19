"""Semantic event-frame lane with independent role and relation choices.

Unlike clause-slot sweeps, this search composes two event frames through a
typed relation (cause, observe, or carry), then checks the resulting sentence
from both character edges.  The right frame is independently selected; no
finished tape or mirrored word list is used to manufacture a hit.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ACTORS = ("aide", "editor", "poet", "teacher", "nurse", "diana")
VERBS = ("reads", "writes", "marks", "helps", "guides", "inspires")
OBJECTS = ("memos", "notes", "poems", "maps", "letters", "men")
PLACES = ("at dawn", "in town", "near home", "by the sea")
RELATIONS = (("because", "causal"), ("while", "contrastive"), ("as", "temporal"))

def norm(s): return normalize_letters(s)

def audit(s):
    t = norm(s); i, j, bad = 0, len(t)-1, []
    while i < j:
        if t[i] != t[j]: bad.append((i, j))
        i += 1; j -= 1
    return {"letters": len(t), "two_pointer_exact": not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def frame(actor, verb, obj, place):
    return f"{actor} {verb} {obj} {place}"

def run(max_rows=5000):
    rows=[]
    for la in ACTORS:
      for lv in VERBS:
       for lo in OBJECTS:
        for lp in PLACES:
         left=frame(la,lv,lo,lp)
         for relation, relation_name in RELATIONS:
          for ra in ACTORS:
           for rv in VERBS:
            for ro in OBJECTS:
             right=frame(ra,rv,ro,lp)
             rendered=f"{left} {relation} {right}."
             a=audit(rendered)
             checks=mechanical_admission_checks(rendered,min_letters=30,max_letters=220)
             rows.append({"rendered":rendered,"left_frame":{"actor":la,"verb":lv,"object":lo,"place":lp},"right_frame":{"actor":ra,"verb":rv,"object":ro,"place":lp},"relation":relation_name,"audit":a,"mechanical_checks":checks,"mechanically_admitted":a["two_pointer_exact"] and all(checks.values()),"reader_status":"not_run; human review required"})
             if len(rows)>=max_rows: break
            if len(rows)>=max_rows: break
           if len(rows)>=max_rows: break
          if len(rows)>=max_rows: break
         if len(rows)>=max_rows: break
        if len(rows)>=max_rows: break
       if len(rows)>=max_rows: break
      if len(rows)>=max_rows: break
    admitted=[r for r in rows if r["mechanically_admitted"]]
    return {"experiment_id":"semantic-event-frame-lane-20260918","signature":"semantic-event-frame|relation-labeled|independent-role-product|two-pointer-sha-audit","config":{"max_rows":max_rows},"stats":{"rows":len(rows),"exact":sum(r["audit"]["two_pointer_exact"] for r in rows),"mechanically_admitted":len(admitted),"longest_letters":max(r["audit"]["letters"] for r in rows)},"rendered_candidates_and_probes":rows,"admitted":admitted,"provenance":{"finished_tape_reversed":False,"catalogue_text_imported":False,"word_order_mirror":False,"independent_validator":"two-pointer normalized tape plus forward/reverse SHA-256","human_readability_certified":False},"next_repair":"add typed argument-selection transitions and independently vary the second frame's setting instead of reusing the first setting","reader_gate":"closed until human intact-prose review"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); ap.add_argument('--max-rows',type=int,default=5000); a=ap.parse_args()
    if a.out.exists(): ap.error(f"refusing to overwrite existing output: {a.out}")
    a.out.parent.mkdir(parents=True,exist_ok=True); result=run(a.max_rows); a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['stats'],indent=2))
if __name__=='__main__': main()
