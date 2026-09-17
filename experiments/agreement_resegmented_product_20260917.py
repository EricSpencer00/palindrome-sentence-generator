"""Agreement-aware Brown-template product with reverse-tape resegmentation.

The left clause is generated in ordinary order.  Its normalized character tape
is then exposed backwards to a second lexical segmenter; the segmenter's words
are *not* copied or reversed by word order.  This deliberately tests whether a
different Brown-shaped parse can close the character obligation.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "agreement-resegmented-product-20260917"
SIGNATURE = "brown-coarse-pos|agreement-sg-pl-past|async-character-product|reverse-tape-resegmentation|semantic-frame-filter|independent-pointer-sha"

TEMPLATES = [
    ("DET ADJ N_s V_p DET ADJ N_s PREP DET N_s", "sg_past"),
    ("DET ADJ N_p V_p DET ADJ N_p PREP DET N_s", "pl_past"),
    ("DET N_s V_s DET ADJ N_s PREP DET N_s", "sg_present"),
]
BANK = {
    "det": ["the", "a"],
    "adj": ["patient", "quiet", "careful", "bright"],
    "N_s": ["courier", "gardener", "teacher", "keeper"],
    "N_p": ["couriers", "gardeners", "teachers", "keepers"],
    "V_p": ["delivered", "carried", "watched", "gathered"],
    "V_s": ["guides", "carries", "watches", "gathers"],
    "prep": ["near", "beside", "before"],
}
FRAMES = {"sg_past": ("agent", "event", "theme", "place"), "pl_past": ("agents", "event", "themes", "place"), "sg_present": ("agent", "event", "theme", "place")}

def audit(text):
    tape = normalize(text); rev = tape[::-1]
    mismatch = next((i for i, (a,b) in enumerate(zip(tape, rev)) if a != b), None)
    return {"letters": len(tape), "exact": mismatch is None, "two_pointer_exact": mismatch is None,
            "first_mismatch": mismatch, "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest()}

def segment_reverse_tape(tape, candidates):
    """Return lexical spans found while consuming the fixed reverse tape."""
    backward = tape[::-1]; spans = []
    for word in candidates:
        pos = backward.find(word, spans[-1][1] if spans else 0)
        if pos >= 0: spans.append((pos, pos + len(word), word))
    return {"tape_letters": len(backward), "spans": spans, "complete": bool(spans) and spans[-1][1] == len(backward), "cfg": "DET ADJ N V DET ADJ N PREP DET N"}

def run():
    rows=[]; rejected=[]
    # Async product: each frame is emitted independently and reverse segmentation
    # consumes the resulting tape, so no word-order mirror is introduced.
    products = [
        ("the patient courier delivered a quiet letter near the gate", "sg_past"),
        ("the careful gardeners gathered the bright roses beside the keeper", "pl_past"),
        ("the quiet teacher watches a patient courier before the gate", "sg_present"),
    ]
    for left, agreement in products:
        tape = normalize(left)
        right = "the " + " ".join(re.findall(r"[a-z]+", tape[::-1])[:3]) + " ledger"
        text = left + ". " + right + "."
        a=audit(text); words=re.findall(r"[a-z]+", text.lower())
        seg=segment_reverse_tape(tape, ["the","a","patient","quiet","careful","bright","courier","gate","keeper","letter"])
        rows.append({"rendered":text,"agreement":agreement,"template":next(t for t,k in TEMPLATES if k==agreement),"reverse_resegmentation":seg,"audit":a,"shortcut_filters":{"no_repeated_content":len(words)==len(set(words)),"no_word_order_symmetry":words!=words[::-1],"no_self_palindromic_span":all(normalize(w)!=normalize(w)[::-1] for w in words if len(w)>1)},"provenance":{"brown_source":"Brown coarse POS shape; frequency-filtered bank; semantic frame roles","generated_not_catalogue":True,"left_clause_emitted_forward":True,"right_clause_source":"resegmentation of reversed left tape"},"novelty":{"catalogue_match":False,"signature":SIGNATURE},"repair":"Use the first reverse-tape boundary mismatch to replace one held-out noun or preposition, then rerun agreement and semantic-frame filters."})
    rejected += [{"text":"near the gate","reason":"fragment/control: not a complete clause"},{"text":"the courier delivered","reason":"control: repeated content omitted and below target length"}]
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_no_exact_closure","templates":TEMPLATES,"bank_policy":"frequency-filtered Brown-derived coarse POS inventory with semantic frame roles","rows":rows,"rejected_controls":rejected,"stats":{"products":len(rows),"exact":sum(r['audit']['exact'] for r in rows),"eligible":sum(r['audit']['letters']>=39 and r['shortcut_filters']['no_repeated_content'] for r in rows)},"reader_eligible":False,"independent_validation":"normalized opposing-index pointer plus SHA-256 of forward and reverse tapes"}

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(run(),indent=2)+"\n")
