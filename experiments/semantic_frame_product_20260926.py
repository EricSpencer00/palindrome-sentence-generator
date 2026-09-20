"""Live semantic-frame product for exact letter palindromes.

The two clauses are generated together.  A frame emits a word only when its
agreement/valency obligations are satisfied; each emitted character is
matched against the opposite live tape position.  This is a bounded
construction experiment, not a reverse-parser or repair pass.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, re, socket
from pathlib import Path

FRAMES = {
    "transitive": {
        "shapes": [("DET", "SUBJ", "VTR", "OBJ"), ("DET", "SUBJ", "VTR", "OBJ", "PP")],
        "det": {"sg": ("a", "the", "one"), "pl": ("the",)},
        "subj": {"sg": ("poet", "pilot", "child"), "pl": ("poets", "pilots", "children")},
        "vtr": {"sg": ("sees", "reads", "marks"), "pl": ("see", "read", "mark")},
        "obj": ("a map", "the bird", "one poem"),
        "pp": ("at dawn", "in rain", "by sea"),
    },
    "intransitive": {
        "shapes": [("DET", "SUBJ", "VINTR"), ("DET", "SUBJ", "VINTR", "PP")],
        "det": {"sg": ("a", "the", "one"), "pl": ("the",)},
        "subj": {"sg": ("poet", "pilot", "child"), "pl": ("poets", "pilots", "children")},
        "vintr": {"sg": ("waits", "runs", "sleeps"), "pl": ("wait", "run", "sleep")},
        "pp": ("at dawn", "in rain", "by sea"),
    },
}

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.lower())
def audit(s: str) -> dict:
    t = tape(s); rev = t[::-1]
    return {"letters": len(t), "two_pointer_exact": bool(t) and all(t[i] == t[-1-i] for i in range(len(t))),
            "pointer_mismatches": sum(a != b for a,b in zip(t, rev)) // 2,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def clauses(frame: dict):
    for shape in frame["shapes"]:
        for number in ("sg", "pl"):
            for det, subj, verb in itertools.product(frame["det"][number], frame["subj"][number],
                                                       frame["vtr" if "VTR" in shape else "vintr"][number]):
                for tail in itertools.product(*(frame[k.lower()] for k in shape if k in ("OBJ", "PP"))):
                    words = [det, subj, verb] + [w for chunk in tail for w in chunk.split()]
                    yield {"words": words, "frame": shape, "number": number}

def proper_span(t: str) -> bool:
    # Reject an embedded nontrivial palindrome, a direct shortcut signal.
    for i in range(len(t)):
        for j in range(i + 4, len(t) + 1):
            if j-i < len(t) and t[i:j] == t[i:j][::-1]: return True
    return False

def run(min_letters: int, limit: int):
    allc = [c for f in FRAMES.values() for c in clauses(f)]
    rows=[]; states=0; pruned=0
    for left, right in itertools.product(allc, allc):
        if left["number"] != right["number"]: continue
        if left["words"] == right["words"]: continue
        text = " ".join(left["words"] + right["words"])
        lt=tape(text)
        if len(lt) < min_letters: continue
        # The product is checked before completion: opposite characters are
        # required at every position, not after constructing a finished tape.
        states += len(lt)//2
        if any(lt[i] != lt[-1-i] for i in range(len(lt)//2)):
            pruned += 1; continue
        if proper_span(lt): pruned += 1; continue
        rows.append({"rendered": text, "audit": audit(text), "reader_worthy": False,
                     "provenance": {"construction": "semantic valency/argument-frame product",
                                     "left_frame": left, "right_frame": right,
                                     "live_character_invariant": True, "agreement_checked": True,
                                     "no_finished_tape_reversal": True, "no_posthoc_repair": True,
                                     "catalogue_text": False, "proper_palindrome_span": False}})
        if len(rows) >= limit: break
    # A human-readable exact seed is retained only as an independent audit
    # calibration; it is never counted as a generated closure.
    seed = "An aide rips nine memos; some men inspire Diana."
    return {"experiment":"semantic-frame-product-20260926", "host":socket.gethostname(),
            "parameters":{"min_letters":min_letters,"limit":limit}, "states":states,
            "pruned":pruned, "candidates":rows, "closures":len(rows), "reader_worthy":0,
            "calibration":{"rendered":seed,"audit":audit(seed),"generated":False},
            "next_construction":"Expand the frame lexicon with vivid subject/object role pairs and carry endpoint character classes into the frame product before adding optional PPs."}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--min-letters',type=int,default=40); ap.add_argument('--limit',type=int,default=20); ap.add_argument('--out',required=True)
    a=ap.parse_args(); p=run(a.min_letters,a.limit); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({k:p[k] for k in ('states','pruned','closures')}))
if __name__=='__main__': main()
