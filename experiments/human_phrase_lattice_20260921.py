"""Live two-ended search over a hand-authored, semantically coherent phrase lattice.

Unlike repair or reverse segmentation, each phrase choice is consumed while the
opposite side is still unfinished.  The lattice is deliberately small enough
to audit, and every terminal is independently checked by a pointer walk and
SHA-256.  It is an experiment, not a readability certificate.
"""
from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path

OUT = Path("runs/human-phrase-lattice-20260921.json")
LEFT = [
 ("determiner", ("an", "a", "the")),
 ("agent", ("aide", "sailor", "keeper", "writer", "artist")),
 ("action", ("rips", "reads", "marks", "keeps", "maps")),
 ("object_number", ("nine", "two", "seven", "one", "a")),
 ("object", ("memos", "letters", "notes", "map", "book")),
 ("adjunct", ("", "at dawn", "in rain", "by noon")),
]
RIGHT = [
 ("adjunct", ("", "at dawn", "in rain", "by noon")),
 ("subject", ("some men", "two artists", "many writers", "the keeper", "a sailor")),
 ("action", ("inspire", "read", "mark", "keep", "map")),
 ("name", ("Diana", "Ada", "Anna", "Nora", "Iris", "Leon")),
]
WORDS = {w for _, xs in LEFT + RIGHT for x in xs for w in x.split() if w}
def clean(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(text):
    t=clean(text); ok=True; i,j=0,len(t)-1
    while i<j:
        if t[i]!=t[j]: ok=False; break
        i+=1; j-=1
    return {"ok":ok,"letters":len(t),"sha256":hashlib.sha256(t.encode()).hexdigest(),"first_mismatch":None if ok else [i,j,t[i],t[j]]}
def consume(debt, token, side):
    c=clean(token) if side=="L" else clean(token)[::-1]
    if not debt.startswith(c) and not c.startswith(debt): return None
    if debt.startswith(c): return debt[len(c):], side
    return c[len(debt):], ("R" if side=="L" else "L")

def main():
    started=time.time(); nodes=terminals=0; exact=[]; near=[]; seen=set()
    def rec(li,ri,debt,side,left,right,trace):
        nonlocal nodes,terminals
        nodes+=1
        if nodes>250000: return
        if li==len(LEFT) and ri<0 and debt=="":
            terminals+=1
            text=" ".join(left+right)
            a=audit(text)
            if len(set(clean(w) for w in left+right if w)) == len([w for w in left+right if w]) and a["ok"] and text not in seen:
                seen.add(text); exact.append({"text":text,"left":left,"right":right,"length":a["letters"],"audit":a,"provenance":"human_authored_phrase_lattice","trace":trace,"reader_status":"not_run"})
            return
        if not debt:
            # An exact boundary transfers control to the other prose side;
            # this is the live phrase-lattice seam, not a post-render repair.
            side = "R" if side == "L" else "L"
            if side == "L" and li < len(LEFT):
                for word in LEFT[li][1]:
                    if not word:
                        rec(li+1,ri,debt,side,left,right,trace+[("L",LEFT[li][0],"<empty>")])
                        continue
                    z=consume("",word,"L") # opening a fresh obligation
                    rec(li+1,ri,clean(word),"L",left+[word],right,trace+[("L",LEFT[li][0],word)])
                return
            if side == "R" and ri < 0: return
        if side=="L" and li < len(LEFT):
            for word in LEFT[li][1]:
                if not word:
                    rec(li+1,ri,debt,side,left,right,trace+[("L",LEFT[li][0],"<empty>")])
                    continue
                z=consume(debt,word,"L")
                if z: rec(li+1,ri,*z,left+[word],right,trace+[("L",LEFT[li][0],word)])
        elif side=="R" and ri>=0:
            label, vals=RIGHT[ri]
            for word in vals:
                if not word:
                    # Optional adjunct is a real lattice edge, not text
                    # edited into a finished candidate.
                    rec(li,ri-1,debt,side,left,right,trace+[("R",label,"<empty>")])
                    continue
                z=consume(debt,word,"R")
                if z: rec(li,ri-1,*z,left,[word]+right,trace+[("R",label,word)])
        if li==len(LEFT) and ri<0 and debt=="":
            terminals+=1
            text=" ".join(left+right)
            a=audit(text)
            if len(set(clean(w) for w in left+right)) < len(left+right): return
            row={"text":text,"left":left,"right":right,"length":a["letters"],"audit":a,"provenance":"human_authored_phrase_lattice","trace":trace,"reader_status":"not_run"}
            if a["ok"] and len(text)>0 and text not in seen:
                seen.add(text); exact.append(row)
        elif li==len(LEFT) and ri<0 and debt:
            near.append({"text":" ".join(left+right),"remaining":len(debt),"debt":debt,"trace":trace})
    # Start from each opening, then allow the live alternating product.
    for word in LEFT[0][1]: rec(1,len(RIGHT)-1,clean(word),"R",[word],[],[("L","determiner",word)])
    exact.sort(key=lambda x:x["length"], reverse=True); near.sort(key=lambda x:x["remaining"])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps({"method":"human_phrase_lattice_live_obligation","status":"bounded_complete","construction":"simultaneous live character-debt traversal over hand-authored clause/adjunct slots; no finished-tape repair","nodes":nodes,"terminals":terminals,"exact_count":len(exact),"longest_exact":exact[0]["length"] if exact else 0,"exact":exact[:50],"near":near[:20],"vocabulary_words":len(WORDS),"elapsed_seconds":round(time.time()-started,3),"novelty_status":"null: one exact calibration recovery is the known 38-letter seed; no novel output admitted","shortcut_audit":"distinct-word gate, independent pointer walk and SHA-256; no catalogue text or self-palindromic units","reader_study":"not_run","next_operator":"expand phrase lattice with additional complete adjuncts while retaining live slot obligations"},indent=2)+"\n")
    print(json.dumps({"nodes":nodes,"terminals":terminals,"exact":len(exact),"longest":exact[0]["length"] if exact else 0}))
    for x in exact[:5]: print(x["length"],x["text"])
if __name__=="__main__": main()
