"""Bounded authored-clause reverse-tape search with a boundary repair operator.

The bank is a small set of newly authored ordinary clauses, used only as a
proposal vocabulary.  Nothing from the bank is reported as a generated
palindrome: a closure must be independently segmented into a second typed
sentence and pass the shared mechanical exclusions.
"""
from __future__ import annotations
import argparse, json, re
from hashlib import sha256
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 39

# Authored proposal material; deliberately ordinary and non-palindromic.
BANK = [
    ("a careful baker opened the old oven", "NP V NP"),
    ("the quiet teacher marked a clear letter", "NP V NP"),
    ("a patient doctor helped the young farmer", "NP V NP"),
    ("our kind writer drafted a brief report", "NP V NP"),
    ("the artist carried a small map", "NP V NP"),
    ("a gentle friend saved the bright story", "NP V NP"),
    ("the gardener cleaned an empty studio", "NP V NP"),
    ("my careful sister found a lost book", "NP V NP"),
    ("the old captain watched a quiet harbor", "NP V NP"),
    ("a young farmer planned the spring garden", "NP V NP"),
    ("the nurse thanked a patient neighbor", "NP V NP"),
    ("our teacher wrote a useful note", "NP V NP"),
    ("the baker fixed an open window", "NP V NP"),
    ("a calm artist painted the blue room", "NP V NP"),
    ("the doctor called a trusted friend", "NP V NP"),
    ("a writer found the hidden answer", "NP V NP"),
]

def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

def lexicon():
    out = {}
    for text, _ in BANK:
        for w in words(text):
            out.setdefault(w, "V" if w in {"opened","marked","helped","drafted","carried","saved","cleaned","found","watched","planned","thanked","wrote","fixed","painted","called"} else ("NP" if w not in {"a","an","the","our","my"} else "DET"))
    return out

def segment(tape: str, lx: dict[str,str], limit=2000):
    """All short lexical segmentations, retaining only NP V NP shape."""
    by_first = {}
    for w, tag in lx.items(): by_first.setdefault(w[0], []).append((w, tag))
    found = []
    def rec(pos, ws, tags):
        if len(found) >= limit: return
        if pos == len(tape):
            if len(ws) >= 5 and "V" in tags and tags[0] in {"DET","NP"}:
                found.append((tuple(ws), tuple(tags)))
            return
        for w, tag in by_first.get(tape[pos], ()):
            if tape.startswith(w, pos):
                rec(pos + len(w), ws + [w], tags + [tag])
    rec(0, [], [])
    return found

def audit(text):
    tape = normalize_letters(text); mismatches=[]
    for i in range(len(tape)//2):
        if tape[i] != tape[-1-i]: mismatches.append([i, len(tape)-1-i, tape[i], tape[-1-i]])
    return {"letters":len(tape), "exact":not mismatches and bool(tape), "mismatches":mismatches[:12], "tape":tape,
            "sha256":sha256(tape.encode()).hexdigest()}

def repair_operator(left: str, lx: dict[str,str]):
    """Mutate one authored content word and maximize reverse segmentation prefix.

    This is a constructive next step after a zero-closure run: it exposes the
    exact boundary character and replacement that blocks a typed closure.
    """
    ws=words(left); vocab=sorted(lx)
    rows=[]
    for i, old in enumerate(ws):
        for new in vocab:
            if new == old or lx[new] != lx[old]: continue
            mutated=ws[:i]+[new]+ws[i+1:]
            tape="".join(mutated)[::-1]
            best=0
            for n in range(1, len(tape)+1):
                if segment(tape[:n], lx, limit=1): best=n
            rows.append({"source":left,"word_index":i,"old":old,"new":new,"segmentable_reverse_prefix":best,"tape":tape})
    return sorted(rows,key=lambda r:(-r["segmentable_reverse_prefix"],r["new"]))[:10]

def run():
    lx=lexicon(); exact=[]; near=[]
    for left,_ in BANK:
        tape=normalize_letters(left)
        for ws,tags in segment(tape[::-1],lx):
            right=" ".join(ws); text=left+"; "+right; a=audit(text)
            gate=mechanical_admission_checks(text,min_letters=MIN_LETTERS,max_letters=240)
            if a["letters"]>=MIN_LETTERS and a["exact"] and all(gate.values()):
                exact.append({"text":text,"audit":a,"left":left,"right":right,"typed_tags":tags,"provenance":"independently authored bank + reverse lexical segmentation","anti_shortcut":gate})
        rev=tape[::-1]; best=0
        for n in range(1,len(rev)+1):
            if segment(rev[:n],lx,limit=1): best=n
        near.append({"left":left,"letters":len(tape),"reverse_prefix_segmentable":best,"reverse_first_mismatch":rev[best:best+12],"repairs":repair_operator(left,lx)[:3]})
    exact.sort(key=lambda x:-x["audit"]["letters"])
    return {"status":"authored_reverse_ngrams_search","config":{"bank_size":len(BANK),"min_letters":MIN_LETTERS,"independent_reverse_segmentation":True,"catalogue_text":False},"candidate_count":len(exact),"candidates":exact[:20],"near_misses":near,"next_operator":"Use the highest-prefix typed repair as a new clause slot, then rerun the independent segmentation; readability still requires blinded human raters.","generator_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); result=run(); a.out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"candidate_count":result["candidate_count"],"bank_size":len(BANK),"best_prefix":max(x["reverse_prefix_segmentable"] for x in result["near_misses"])},indent=2))
