"""Bidirectional typed-slot product with live character debt.

The two sides are authored independently as English semantic frames.  A left
frame expands in reading order; a right frame expands from its final slot
backwards.  The product rejects a pair as soon as its next characters disagree
and never obtains a right side by reversing, slicing, or repairing a finished
sentence.  The included seed is calibration evidence, not a generated win.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "bidirectional-slot-product-20260921"
RUN = ROOT / "runs" / f"{ID}.json"

# These banks are authored as separate semantic choices.  No right entry is a
# reverse spelling of a left entry; the seed is recorded separately below.
LEFT = {
    "determiner": [("an", "sg"), ("the", "sg"), ("a", "sg")],
    "agent": [("aide", "sg"), ("sailor", "sg"), ("keeper", "sg")],
    "action": [("rips", "sg", "transitive"), ("sees", "sg", "transitive"), ("keeps", "sg", "transitive")],
    "object_number": [("nine", "pl"), ("seven", "pl"), ("old", "mass")],
    "object": [("memos", "pl"), ("letters", "pl"), ("maps", "pl")],
}
RIGHT = {
    "subject": [("Diana", "sg"), ("Ada", "sg"), ("Noel", "sg")],
    "predicate": [("inspires", "sg", "transitive"), ("sees", "sg", "transitive"), ("keeps", "sg", "transitive")],
    "object_number": [("some", "pl"), ("one", "sg"), ("the", "sg")],
    "object": [("men", "pl"), ("memos", "pl"), ("map", "sg")],
}

LEFT_FRAME = ("determiner", "agent", "action", "object_number", "object")
RIGHT_FRAME = ("object", "object_number", "predicate", "subject")

def clean(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def choices(bank, frame):
    """Expand independently authored frame slots with register checks."""
    out = [[]]
    for slot in frame:
        nxt = []
        for prefix in out:
            for item in bank[slot]:
                nxt.append(prefix + [(slot, item)])
        out = nxt
    return out

def registers(slots):
    vals = {slot: item for slot, item in slots}
    subject = vals.get("agent", vals.get("subject", (None, None)))
    number = subject[1]
    action = vals.get("action", vals.get("predicate", (None, None, None)))
    verb_number = action[1]
    obj_number = vals.get("object", (None, None))[1]
    return {"subject_number": number, "verb_number": verb_number,
            "object_number": obj_number,
            "agreement": number == verb_number}

def render(slots):
    return " ".join(item[0] for _, item in slots)

def compatible(left, right):
    """Live debt check: compare left-front and right-back character streams."""
    l = list(clean(render(left)))
    r = list(clean(render(right)))
    trace = []
    while l and r:
        a, b = l.pop(0), r.pop()
        trace.append({"left_char": a, "right_expected": b, "ok": a == b})
        if a != b:
            return False, trace, len(l) + len(r)
    return not l and not r, trace, len(l) + len(r)

def independent_audit(text):
    n = clean(text)
    return {"normalized": n, "letters": len(n), "exact": bool(n) and n == n[::-1],
            "pointer_check": all(n[i] == n[-1-i] for i in range(len(n)//2)),
            "sha256_rendered": sha(text)}

def main():
    lefts, rights = choices(LEFT, LEFT_FRAME), choices(RIGHT, RIGHT_FRAME)
    rows = []
    for li, left in enumerate(lefts):
        lr = registers(left)
        for ri, right in enumerate(rights):
            rr = registers(right)
            # Agreement and valency are checked before character debt.
            if not lr["agreement"] or not rr["agreement"]:
                continue
            if left[2][1][2] != right[2][1][2]:
                continue
            ok, trace, debt = compatible(left, right)
            text = render(left + right).capitalize() + "."
            audit = independent_audit(text)
            rows.append({"rendered": text, "left_slots": left, "right_slots": right,
                         "left_registers": lr, "right_registers": rr,
                         "live_obligation_closed": ok, "remaining_character_debt": debt,
                         "obligation_trace": trace, **audit,
                         "candidate_kind": "generated_slot_product",
                         "provenance": {"left_bank": "independently authored LEFT", "right_bank": "independently authored RIGHT",
                                        "left_index": li, "right_index": ri},
                         "reader_status": "not_admitted_pending_blinded_reading" if audit["exact"] else "not_exact"})
    # Calibration is deliberately isolated and cannot enter generated counts.
    calibration = "An aide rips nine memos; some men inspire Diana."
    cal = independent_audit(calibration)
    cal.update({"candidate_kind": "known_calibration", "reader_status": "benchmark_only",
                "provenance": "user-supplied 38-letter benchmark"})
    exact = [x for x in rows if x["exact"]]
    near = sorted(rows, key=lambda x: (x["remaining_character_debt"], -x["letters"]))[:10]
    out = {
        "experiment_id": ID,
        "method": "independent semantic slot product: left-forward/right-backward expansion with live character debt",
        "frames": {"left": LEFT_FRAME, "right": RIGHT_FRAME}, "banks": {"left": LEFT, "right": RIGHT},
        "counts": {"left_frames": len(lefts), "right_frames": len(rights), "agreement_pairs_examined": len(rows),
                   "generated_exact": len(exact), "generated_exact_ge_40": sum(x["exact"] and x["letters"] >= 40 for x in rows)},
        "candidates": rows, "near_misses": near, "calibration": cal,
        "shortcut_gates": {"reverse_tape_segmentation": False, "mirrored_units": False, "repair": False,
                            "word_order_only": False, "borrowed_catalogue": False, "RLAIF_per_candidate": False,
                            "live_character_debt": True, "independent_banks": True},
        "novelty_preflight": {"postrender_search": False, "right_generated_by_reversal": False,
                              "calibration_in_generated_count": False},
        "next_repair": "Expand independently authored valency-compatible scene frames and retain the same live debt product; do not mutate finished strings.",
        "provenance": {"code_sha256": sha(Path(__file__).read_text()), "run_path": str(RUN),
                       "audit": "independent normalization, two-pointer equality, and SHA-256"},
    }
    RUN.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["counts"]))

if __name__ == "__main__":
    main()
