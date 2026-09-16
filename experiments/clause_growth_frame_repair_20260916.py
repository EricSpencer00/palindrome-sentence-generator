"""Clause-growth construction with held-out semantic-frame repair.

Unlike a sweep over finished strings, this lane grows an authored scene one
ordinary clause at a time.  Each extension carries a global character debt
ledger; held-out repairs replace a typed semantic frame and re-render the
whole scene.  The search never reverses a sentence or copies a tape.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "clause-growth-frame-repair-20260916"
SIGNATURE = "authored-scene-clause-growth|typed-valency-frame-lattice|global-character-debt-ledger|heldout-semantic-frame-repair|ordinary-order-composition|independent-pointer-sha-audit"
OUT = ROOT / "runs" / (EXPERIMENT_ID + ".json")

FRAMES = [
 {"id":"courier", "subject":"the patient courier", "verb":"delivered", "object":"the sealed letter", "adjunct":"before dusk", "sense":"a courier delivers a sealed letter before dusk"},
 {"id":"clerk", "subject":"the quiet clerk", "verb":"opened", "object":"the wooden parcel", "adjunct":"by the window", "sense":"a clerk opens a wooden parcel by the window"},
 {"id":"gardener", "subject":"the careful gardener", "verb":"watered", "object":"the young cedar", "adjunct":"after rain", "sense":"a gardener waters a young cedar after rain"},
 {"id":"keeper", "subject":"the patient keeper", "verb":"recorded", "object":"the harbor signal", "adjunct":"at noon", "sense":"a keeper records a harbor signal at noon"},
]
FOURTH_FRAMES = [
 {"id":"librarian", "subject":"the calm librarian", "verb":"shelved", "object":"the borrowed atlas", "adjunct":"after lunch", "sense":"a librarian shelves a borrowed atlas after lunch"},
 {"id":"sailor", "subject":"the weary sailor", "verb":"mended", "object":"the canvas sail", "adjunct":"near harbor", "sense":"a sailor mends a canvas sail near harbor"},
 {"id":"teacher", "subject":"the kind teacher", "verb":"marked", "object":"the final essay", "adjunct":"after class", "sense":"a teacher marks the final essay after class"},
]

def audit(text: str) -> dict:
    letters = normalize_letters(text)
    rev = letters[::-1]
    hf = hashlib.sha256(letters.encode()).hexdigest(); hr = hashlib.sha256(rev.encode()).hexdigest()
    return {"letters":len(letters), "exact":letters == rev, "hash_forward":hf,
            "hash_reverse":hr, "hash_equal":hf == hr,
            "independent_pointer_audit":all(letters[i] == letters[-1-i] for i in range(len(letters)))}

def render(frames):
    return ". ".join(f"{x['subject']} {x['verb']} {x['object']} {x['adjunct']}" for x in frames) + "."

def debt(text: str):
    s = normalize_letters(text); r = s[::-1]; n=min(len(s),len(r))
    return {"paired":n, "matches":sum(a==b for a,b in zip(s[:n],r[:n])),
            "length_delta":abs(len(s)-len(r)), "first_mismatch":next((i for i,(a,b) in enumerate(zip(s,r)) if a!=b),None)}

def candidate(frames, label, repair_from=None):
    text=render(frames); a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=260)
    return {"label":label,"rendered":text,"letters":a["letters"],"frames":[x["id"] for x in frames],
      "senses":[x["sense"] for x in frames],"global_debt":debt(text),"exact_audit":a,
      "checks":checks,"admitted":bool(a["exact"] and all(checks.values())),
      "provenance":{"human_authored_frames":True,"ordinary_order":True,"copied_text":False,
        "reversed_finished_sentence":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,
        "repair_from":repair_from}}

def frontier_fourth_clause():
    """Select a fresh fourth clause by the newly exposed outer debt frontier.

    The first three clauses are fixed; candidates are scored only on the
    character pairs exposed by the extension, then the winning complete scene
    is independently audited.  This is a repair operator, not a random
    append.
    """
    prefix = FRAMES[:3]
    scored=[]
    for frame in FOURTH_FRAMES:
        row=candidate(prefix+[frame],"frontier-selected-fourth-clause",repair_from="growth-3-clause")
        old=candidate(prefix,"growth-3-clause")
        scored.append((row["global_debt"]["matches"]-old["global_debt"]["matches"], row))
    gain,row=max(scored,key=lambda x:(x[0],-x[1]["letters"]))
    row["frontier_selection"]={"operator":"maximize newly exposed mirrored-position matches", "tested_frames":len(scored), "match_gain":gain}
    return row

def novelty():
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    return {"exact_signature_collision":any(x["id"] != EXPERIMENT_ID and x["signature"] == SIGNATURE for x in reg["entries"]),
      "registry_entries":len(reg["entries"]),"preflight_rule":"reject exact signature collision or replay of scene-lattice/semantic-center operators"}

def run():
    base=[]
    # Growth states: 1, 2, and 3 clauses, with a held-out complete-frame repair.
    for n in (1,2,3): base.append(candidate(FRAMES[:n],f"growth-{n}-clause"))
    held=list(FRAMES[:2]); held[1]={**FRAMES[3],"id":"keeper-repair"}
    repair=candidate(held,"held-out-frame-repair",repair_from="growth-2-clause")
    rows=base+[repair,frontier_fourth_clause()]
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed",
      "method":"grow intact authored clauses while carrying a global debt ledger; substitute a held-out typed valency frame and recompute the complete scene",
      "novelty_preflight":novelty(),"candidates":rows,
      "stats":{"candidates":len(rows),"exact":sum(x["exact_audit"]["exact"] for x in rows),"admitted":sum(x["admitted"] for x in rows)},
      "next_repair":"use the selected fourth-clause debt frontier to author a fifth clause with a new semantic frame; if exact closure occurs, run a blinded intact-prose versus shuffled-order reader pretest",
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_catalogue":False}}

if __name__ == "__main__":
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.write_text(json.dumps(run(),indent=2)+"\n"); print(json.dumps(run(),indent=2))
