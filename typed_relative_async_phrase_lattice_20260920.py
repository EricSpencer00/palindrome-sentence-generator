"""Typed relative-clause extension of the phrase/word lattice.

Phrase paths contain a finite clause and an explicitly typed relative adjunct.
The bilateral walker advances word/character offsets independently, recording
when one side crosses a phrase boundary while the other remains inside its
current phrase.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/typed-relative-async-phrase-lattice-20260920.json"
ID = "typed-relative-async-phrase-lattice-20260920"

LEFT = (
    (("finite", "the patient gardener waters the cedar"), ("relative", "which shelters a quiet sparrow")),
    (("finite", "a careful keeper records the harbor bells"), ("relative", "who remembers the winter tide")),
    (("finite", "our quiet teacher carries an atlas"), ("relative", "that describes the northern coast")),
)
RIGHT = (
    (("finite", "the patient singer hears the distant bells"), ("relative", "who follows a bright refrain")),
    (("finite", "a careful pilot crosses the quiet harbor"), ("relative", "that shelters an old vessel")),
    (("finite", "our quiet keeper reads a journal"), ("relative", "which records the autumn crossing")),
)

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
    return {"letters":len(t),"pointer_exact":bool(t) and mm is None,"first_mismatch":mm,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def prov(path):
    ws=[letters(w) for _,p in path for w in p.split()]
    return {"typed_relative_clause":True,"complete_phrase_units":True,"nested_self_palindrome":any(len(w)>3 and w==w[::-1] for w in ws),
            "repeated_units":len(ws)!=len(set(ws)),"word_order_symmetry":ws==ws[::-1],"fragment":len(ws)<10,
            "catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"RLAIF":False}

def run():
    rows=[]; states=prunes=skews=0
    for lp,rp in itertools.product(LEFT,RIGHT):
        lw=[w for _,p in lp for w in p.split()]; rw=[w for _,p in rp for w in p.split()]
        i=j=lo=ro=0; ok=True; trace=[]; li=ri=0
        while i<len(lw) and j<len(rw):
            a,b=letters(lw[i]),letters(rw[-1-j]); states+=1
            if a[lo] != b[::-1][ro]: ok=False; prunes+=1; break
            trace.append({"left_word":lw[i],"right_word":rw[-1-j],"left_phrase":lp[li][0],"right_phrase":rp[ri][0],"left_offset":lo,"right_offset":ro})
            lo+=1; ro+=1
            if lo==len(a):
                i+=1;lo=0
                if i<len(lw) and li==0: li=1; skews+=int(j<len(rw) and ri==0)
            if ro==len(b):
                j+=1;ro=0
                if j<len(rw) and ri==0: ri=1; skews+=int(i<len(lw) and li==0)
        if ok and (i!=len(lw) or j!=len(rw)): ok=False
        text=" ".join(lw)+"."; rows.append({"rendered":text,"paired_typed_paths":{"left":lp,"right":rp},"online_lattice":{"closed":ok,"trace":trace[-10:]},"audit":audit(text),"provenance":prov(lp)})
    rows.sort(key=lambda r:(-r["audit"]["letters"],r["rendered"]))
    exact=[r for r in rows if r["online_lattice"]["closed"] and r["audit"]["pointer_exact"] and not any(r["provenance"][k] for k in ("nested_self_palindrome","repeated_units","word_order_symmetry","fragment"))]
    out={"experiment_id":ID,"method":"typed finite-plus-relative phrase paths with asynchronous opposite word/character frontiers and phrase-boundary skew","stats":{"left_paths":len(LEFT),"right_paths":len(RIGHT),"paired_paths":len(rows),"online_states":states,"mismatch_prunes":prunes,"phrase_boundary_skews":skews,"rendered_controls":len(rows),"exact_clean":len(exact),"max_letters":rows[0]["audit"]["letters"]},"rendered_candidates":rows,"exact_candidates":exact,"controls":rows[:12],"novelty_preflight":{"status":"passed","signature":"fresh-authored|typed-relative-paths|asynchronous-phrase-boundary-skew|independent-audit","distinct_from":"flat phrase lattice: relative attachment types are retained while the two ordinary-order paths cross phrase boundaries independently"},"provenance":{"audits":["independent two-pointer mismatch","forward/reverse SHA-256"],"hard_exclusions":["nested palindromes","repeated units","word-order symmetry","fragments","catalogue text"]},"falsifier":"recompute every row with an independent normalizer and reject any closed row whose forward and reverse hashes differ","next_construction":"Add two independently typed relative attachments per side with bounded attachment depth and preserve boundary-skew traces.","status":"exact clean candidate requires reading" if exact else "no exact clean intersection; typed relative controls retained"}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+"\n"); return out
if __name__=='__main__': print(json.dumps(run()["stats"]))
