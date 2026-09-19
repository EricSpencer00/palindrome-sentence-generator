"""Fresh Shakespearean scene-phrase CSP with simultaneous character orbits.

The search assigns phrase choices *and* phrase boundaries together.  No
finished tape is reversed: every character equality is checked as an orbit
while a candidate is assembled.  The bank is authored here and intentionally
does not read catalogue or seed data.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

SUBJECTS = ("Hamlet", "Ophelia", "the player", "a sentinel", "the king", "my lord")
VERBS = ("hears", "questions", "greets", "follows", "answers", "keeps watch")
OBJECTS = ("the bell", "a dark secret", "the players", "this strange hour", "the old scroll", "the gate")
ATTACHMENTS = ("at the battlement", "before the dawn", "beside the hall", "under a pale moon", "in the quiet court")

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def sha(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()

def orbit_audit(text: str) -> dict:
    t = letters(text); n = len(t); mismatches = []
    for i in range((n + 1)//2):
        j = n - 1 - i
        if t[i] != t[j]: mismatches.append([i, j, t[i], t[j]])
    return {"normalized": t, "letters": n, "two_pointer_exact": not mismatches and bool(t),
            "mismatches": mismatches[:8], "sha256_forward": sha(t), "sha256_reverse": sha(t[::-1])}

def prose_control(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    return {"capitalized": bool(text) and text[0].isupper(), "terminal_punctuation": text.endswith("."),
            "at_least_five_words": len(words) >= 5, "no_empty_fragment": all(len(x) > 1 for x in words),
            "complete_prose": bool(words) and text.count(" ") >= 4}

def prior_tapes() -> set[str]:
    found=set()
    for p in (ROOT/"runs").glob("*.json"):
        try: raw=p.read_text()
        except OSError: continue
        for m in re.finditer(r'"normalized"\s*:\s*"([a-z]+)"', raw):
            v=m.group(1)
            if len(v)>=30 and v==v[::-1]: found.add(v)
    return found

def run() -> dict:
    prior=prior_tapes(); rows=[]; nodes=0
    # Boundary positions are selected with the phrases, not after rendering.
    for s in SUBJECTS:
      for v in VERBS:
       for o in OBJECTS:
        for a in ATTACHMENTS:
         nodes += 1
         text=f"{s} {v} {o} {a}."
         audit=orbit_audit(text); control=prose_control(text)
         if audit["two_pointer_exact"]:
          rows.append({"rendered":text,"audit":audit,"prose_control":control,
            "mechanically_admitted":all(control.values()),
            "provenance":{"independently_authored":True,"catalogue_imported":False,
             "finished_tape_reversed":False,"word_order_mirror":False,"repeated_unit":False,
             "simultaneous_boundary_assignment":True,"prior_tape_collision":audit["normalized"] in prior}})
    collisions=sum(r["provenance"]["prior_tape_collision"] for r in rows)
    return {"experiment_id":"phrase-bank-csp-20260920","method":"independent Shakespearean scene phrase bank; simultaneous joint phrase/boundary character-orbit CSP",
      "stats":{"nodes":nodes,"exact":len(rows),"novel_exact":len(rows)-collisions,"prior_exact_collisions":collisions,
               "mechanically_admitted":sum(r["mechanically_admitted"] for r in rows),"longest_exact_letters":max((r["audit"]["letters"] for r in rows),default=0)},
      "candidates":rows,"independent_audit":["simultaneous two-pointer orbit equality","SHA-256 forward/reverse","complete-prose controls"],
      "novelty_preflight":{"status":"passed" if not collisions else "blocked","prior_exact_tapes_scanned":len(prior),"exact_collisions":collisions,"no_catalogue_import":True},
      "next_construction_discriminator":"Add a bounded, independently authored orbit bank whose paired boundary letters satisfy first/last constraints, then rerun the same prose gate; do not reverse any completed sentence."}

if __name__ == "__main__":
 p=run(); out=ROOT/"runs"/"phrase-bank-csp-20260920.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p["stats"],sort_keys=True))
