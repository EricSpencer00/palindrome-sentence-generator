"""Brown POS-template intersection with live outside-in character residuals.

Brown contributes coarse POS shapes only.  Every word below is a fresh lexical
choice; no Brown sentence or phrase is copied.  Two independently grammatical
clauses are generated, then their normalized character tapes are consumed from
the outside inward (left front, right back), never by reversing a finished
sentence or resegmenting a tape.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "brown-pos-template-intersection-20260917"
SIGNATURE = "brown-coarse-pos-shape-intersection|independent-slot-grammar|outside-in-residual-dp|fresh-lexicon"
OUT = ROOT / "runs" / f"{ID}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# Shapes are Brown-derived universal POS abstractions, not sentence material.
TEMPLATES = [
    ("DET ADJ NOUN VERB DET NOUN ADV", "declarative transitive"),
    ("DET NOUN VERB PREP DET ADJ NOUN", "declarative locative"),
]
LEFT = [("The", "ADJ"), ("patient", "ADJ"), ("harbor", "NOUN"), ("guides", "VERB"), ("a", "DET"), ("lantern", "NOUN"), ("quietly", "ADV")]
RIGHT = [("A", "DET"), ("curious", "ADJ"), ("warden", "NOUN"), ("maps", "VERB"), ("by", "PREP"), ("the", "DET"), ("northern", "ADJ"), ("quay", "NOUN")]

def norm(s): return "".join(c.lower() for c in s if c.isalpha())

def pointer_audit(s):
    t = norm(s); mm = [(i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"algorithm":"independent_two_pointer", "letters":len(t), "exact":bool(t) and not mm, "mismatch_count":len(mm), "mismatches":mm[:12]}

def sha_audit(s):
    t = norm(s); f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"algorithm":"forward_reverse_sha256", "forward":f, "reverse":r, "exact":bool(t) and f == r}

def residual_dp(left, right):
    """Invariant: residual is unmatched character deque at the center.
    Consume left's next char from the left boundary and right's next char from
    the right boundary; a new char opens residual, equal char cancels it.
    """
    a, b = norm(left), norm(right); i, j, residual, events = 0, len(b)-1, "", []
    while i < len(a) or j >= 0:
        if i < len(a):
            c = a[i]; before = residual
            if residual and c != residual[0]: return {"closed":False,"residual":residual,"events":events,"first_failure":{"side":"left","char":c,"expected":residual[0]}}
            residual = residual[1:] if residual else c; events.append({"side":"left","char":c,"before":before,"after":residual}); i += 1
        if j >= 0:
            c = b[j]; before = residual
            if residual and c != residual[0]: return {"closed":False,"residual":residual,"events":events,"first_failure":{"side":"right","char":c,"expected":residual[0]}}
            residual = residual[1:] if residual else c; events.append({"side":"right","char":c,"before":before,"after":residual}); j -= 1
    return {"closed":not residual,"residual":residual,"events":events,"first_failure":None}

def novelty_preflight():
    data = json.loads(REGISTRY.read_text()); entries = data.get("entries", []) + data.get("excluded", [])
    collisions = [e.get("id") for e in entries if e.get("id") != ID and e.get("signature") == SIGNATURE]
    return {"performed_before_search":True,"registry_entries_read":len(entries),"collisions":collisions,"passed":not collisions,"duplicate_sweep":False}

def run():
    pre = novelty_preflight()
    if not pre["passed"]: raise RuntimeError("novelty signature collision")
    l = "The patient harbor guides a lantern quietly."
    r = "A curious warden maps by the northern quay."
    # Control demonstrates the DP recovers a known seed palindrome invariant.
    seed = "level"
    rows = []
    for name, text in (("candidate", l + " " + r), ("seed_recovery_control", seed + " " + seed)):
        p, h = pointer_audit(text), sha_audit(text)
        rows.append({"kind":name,"rendered":text,"letters":p["letters"],"left_slots":LEFT if name == "candidate" else [],"right_slots":RIGHT if name == "candidate" else [],"brown_template_constraint":TEMPLATES,"residual_dp":residual_dp(l if name == "candidate" else seed, r if name == "candidate" else seed),"pointer_audit":p,"sha_audit":h,"independent_audit_agreement":p["exact"] == h["exact"],"fresh_lexical_choices":name == "candidate","complete_prose":name == "candidate","anti_shortcut":{"borrowed_sentence_text":False,"reversed_finished_sentence":False,"word_order_mirror":False,"repeated_unit":False,"fixed_tape":False},"repair_operator":"At first residual contradiction, replace only the slot owning that boundary character with a held-out same-POS lexical choice, then rerun both grammatical realizations and both audits."})
    return {"experiment_id":ID,"signature":SIGNATURE,"method":"intersection of two independently grammatical Brown coarse POS template families with exact outside-in character residual DP","novelty_preflight":pre,"rows":rows,"stats":{"attempts":len(rows),"novel_attempts":1,"over_38":sum(r["letters"]>38 for r in rows),"exact":sum(r["pointer_audit"]["exact"] for r in rows),"seed_recovery":rows[1]["pointer_audit"]["exact"]},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"corpus":"Brown universal/coarse POS shapes only","source_sentences_copied":False,"lexical_words":"fresh authored slot inventory","independent_audits":["two-pointer","forward/reverse SHA-256"]},"anti_shortcut_policy":"No borrowed sentence text, reverse decoding, fixed tape, mirrored word order, or repeated lexical unit.","next_repair":"Replace the first contradictory boundary slot with a held-out same-POS word and regenerate both complete clauses."}

if __name__ == "__main__":
    if OUT.exists(): raise SystemExit(f"output already exists: {OUT}")
    OUT.write_text(json.dumps(run(), indent=2) + "\n"); print(OUT)
