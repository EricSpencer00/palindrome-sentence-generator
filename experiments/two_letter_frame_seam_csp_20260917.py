"""Bounded two-letter seam CSP over authored relative-clause alternatives."""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "two-letter-frame-seam-csp-20260917"
LEFT = [("the harbor pilot", "studies", "the coastal chart", "that guides the crew"),
        ("the village teacher", "carries", "a weathered map", "which charts the shore"),
        ("the patient keeper", "marks", "the tide ledger", "that records the route"),
        ("the old gardener", "tends", "a cedar board", "which guards the pier")]
RIGHT = [("the quiet sailor", "reads", "the tide ledger", "that remembers the route"),
         ("the patient guide", "keeps", "a field journal", "which records the way"),
         ("a careful captain", "follows", "a tide book", "that watches the quay"),
         ("the young scout", "copies", "a small log", "which notes the paths")]

def letters(text): return re.sub(r"[^a-z]", "", text.lower())
def audit(text):
    tape=letters(text); mismatches=sum(a!=b for a,b in zip(tape,tape[::-1]))
    return {"letters":len(tape),"two_pointer_exact":bool(tape) and mismatches==0,"mismatches":mismatches,
            "sha256_forward":hashlib.sha256(tape.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(tape[::-1].encode()).hexdigest()}

def run():
    rows=[]; rejected=0
    for li,(_,_,_,lr) in enumerate(LEFT):
        for ri,(rs,rv,ro,rr) in enumerate(RIGHT):
            left_sig=letters(lr)[:2]; right_sig=letters(rr)[:2]
            if left_sig != right_sig:
                rejected += 1; continue
            ls,lv,lo,_=LEFT[li]
            text=f"At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet."
            rows.append({"left_slot":li,"right_slot":ri,"rendered":text,
                         "csp":{"left_signature":left_sig,"right_signature":right_sig,"satisfied":True},
                         "audit":audit(text),
                         "provenance":{"pre_render_csp":True,"two_letter_signature":True,
                                        "source_experiment":"frame-pair-seam-csp-20260917","catalogue_used":False,
                                        "wrapped_seed":False,"finished_tape_reversal":False,"word_order_only_symmetry":False}})
    best=min(rows,key=lambda r:r["audit"]["mismatches"])
    return {"experiment":EXPERIMENT,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rendered_candidates":rows,
            "stats":{"rendered":len(rows),"rejected_pre_render":rejected,"exact":sum(r["audit"]["two_pointer_exact"] for r in rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"best_mismatches":best["audit"]["mismatches"]},
            "next_repair":{"operator":"three-letter seam signature with inflectional agreement","reason":"two-letter local compatibility admits grammatical prose but does not close the global tape; next add a bounded morphology-aware signature","route_exhausted":False},
            "provenance":{"bounded_left_slots":len(LEFT),"bounded_right_slots":len(RIGHT),"rejected_before_render":rejected,"catalogue_used":False}}

if __name__=='__main__':
    result=run()
    for directory in (ROOT/'runs',ROOT/'artifacts'):
        (directory/f'{EXPERIMENT}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['stats'],sort_keys=True))
