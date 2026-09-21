"""Online bilateral CFG character intersection (no finished-tape reversal).

Each chart state contains two unfinished CFG stacks.  The left stack expands in
reading order; the right stack is initialized with the reverse RHS of a normal
CFG derivation and expands from the sentence end.  Terminals consume a live
character debt, so no completed candidate is used to manufacture its mirror.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/live-cfg-character-intersection-20260921.json"

LEX = {
    "Det": ("an", "a", "the", "some"),
    "Agent": ("aide", "sailor", "keeper", "artist", "writer", "pilot"),
    "Verb": ("rips", "sees", "keeps", "marks", "reads", "helps", "writes"),
    "Num": ("nine", "seven", "one", "two"),
    "Obj": ("memos", "letters", "maps", "notes", "books", "boats", "gate"),
    "Person": ("men", "women", "sailors", "artists", "writers", "pilots", "poets"),
    "Pred": ("inspire", "read", "see", "help", "mark", "write", "keep"),
    "Name": ("Diana", "Ada", "Anna", "Nora", "Mira", "Iris", "Leon", "Noah"),
}
LEX.update({"Det2": LEX["Det"], "Person": LEX["Person"], "Pred": LEX["Pred"], "Name": LEX["Name"]})
GRAM = {
    "S": (("NP", "VP"),), "NP": (("Det", "Agent"),),
    "VP": (("Verb", "Num", "Obj"),),
    "T": (("NP2", "VP2"),), "NP2": (("Det2", "Person"),),
    "VP2": (("Pred", "Name"),),
}
TERMS = set(LEX) | {"Det2", "Person", "Pred", "Name"}

def clean(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = clean(s); rev = t[::-1]
    return {"normalized": t, "letters": len(t), "exact": bool(t) and t == rev,
            "pointer_check": bool(t) and all(t[i] == t[-1-i] for i in range(len(t)//2)),
            "sha256_normalized": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}
def consume(debt, token, side):
    chars = clean(token) if side == "L" else clean(token)[::-1]
    if debt.startswith(chars): return debt[len(chars):], side
    if chars.startswith(debt): return chars[len(debt):], "L" if side == "R" else "R"
    return None
def shortcut(s):
    ws = re.findall(r"[a-z]+", s.casefold())
    return (["self_palindromic_word_unit"] if any(len(w)>1 and w==w[::-1] for w in ws) else []) + (["repeated_word_unit"] if len(ws)!=len(set(ws)) else [])
def expand(stack, reverse=False):
    sym = stack[0]
    if sym not in GRAM: return [(stack[1:], sym)]
    out=[]
    for rhs in GRAM[sym]: out.append((tuple(reversed(rhs))+stack[1:] if reverse else rhs+stack[1:], None))
    return out

def main():
    # The right grammar is ordinary forward CFG T, traversed by reverse RHS.
    nodes=0; terminals=0; prunes=0; exact=[]; controls=[]; seen=set()
    def rec(ls, rs, debt, side, lw, rw, trace):
        nonlocal nodes, terminals, prunes
        nodes += 1
        if nodes > 400000: return
        key=(ls,rs,debt,side)
        if key in seen: return
        seen.add(key)
        if not ls and not rs:
            terminals += 1
            if not debt:
                text=" ".join(lw)+"; "+" ".join(reversed(rw))
                a=audit(text); row={"text":text,"audit":a,"shortcut_reasons":shortcut(text),"provenance":"online_cfg_terminal_expansions","trace":trace}
                if not row["shortcut_reasons"]: exact.append(row)
            return
        if side=="L" and not ls:
            side="R"
        if side=="R" and not rs:
            side="L"
        stack = ls if side=="L" else rs
        if not stack: return
        for newstack, term in expand(stack, reverse=(side=="R")):
            if term is None:
                rec(newstack,rs,debt,side,lw,rw,trace) if side=="L" else rec(ls,newstack,debt,side,lw,rw,trace)
                continue
            vals=LEX[term]
            for word in vals:
                # The first left terminal opens the obligation; subsequent
                # terminals consume alternately from the live debt.
                got=(clean(word), "R") if side=="L" and not debt else consume(debt,word,side)
                if got is None: prunes += 1; continue
                nd,ns=got
                if side=="L": rec(newstack,rs,nd,ns,lw+[word],rw,trace+[("L",term,word,nd)])
                else: rec(ls,newstack,nd,ns,lw,rw+[word],trace+[("R",term,word,nd)])
    # Start with ordinary S on the left and ordinary T on the right.
    rec(("S",),("T",),"","L",[],[],[])
    exact.sort(key=lambda x:x["audit"]["letters"], reverse=True)
    out={"experiment_id":"live-cfg-character-intersection-20260921","status":"bounded_online_chart","method":"bilateral packed CFG stacks with live character debt","stats":{"nodes":nodes,"unique_states":len(seen),"terminal_derivations":terminals,"character_prunes":prunes,"exact_shortcut_clean":len(exact)},"candidates":exact[:20],"controls":controls,"shortcut_gates":{"finished_tape_reverse":False,"catalogue_text":False,"repair":False,"rlaif":False,"word_order_symmetry":False,"reader_gate":"closed; no human evidence"},"provenance":{"source":"hand-authored finite CFG lexical banks","independent_verifier":"two-pointer plus SHA-256 forward/reverse","next_operator":"add typed adjunct production only at the highest-scoring live chart frontier"}}
    RUN.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["stats"],indent=2)); print(json.dumps(exact[:3],indent=2))
if __name__ == "__main__": main()
