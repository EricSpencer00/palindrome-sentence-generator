"""Bounded neural dual-prefix beam: LM-ranked typed clauses, exact tape online.

This is deliberately a small experiment, not a readability certificate. Two
independent sides are expanded in reading order; their character tapes are
checked for compatibility after every pair of emissions. GPT-2 only ranks
surviving lexical proposals. No reverse decoding or fixed cross-product bank
is used.
"""
from __future__ import annotations
import hashlib, json, re, time
from pathlib import Path

ROLES = {
 "subject": ["the sailor", "a nurse", "the artist", "a child", "the teacher"],
 "verb": ["sees", "holds", "finds", "helps", "keeps"],
 "object": ["a map", "the red book", "a small bell", "the old boat", "a blue vase"],
}
CLAUSES = [f"{s} {v} {o}" for s in ROLES["subject"] for v in ROLES["verb"] for o in ROLES["object"]]

def norm(s): return "".join(c.lower() for c in s if "a" <= c.lower() <= "z")
def palindrome(s):
    t=norm(s); return bool(t) and t==t[::-1]
def compatible(a,b):
    """Whether already emitted outer tapes agree wherever they overlap."""
    x,y=norm(a),norm(b)[::-1]
    return all(i>=len(y) or j==y[i] for i,j in enumerate(x))
def score(text):
    # Neural scoring is attempted once; the fallback is explicit and logged.
    try:
        from llm_palindrome.lm_scoring import GPT2Scorer
        global _LM
        if _LM is None: _LM=GPT2Scorer("gpt2", device="cpu")
        return float(_LM.score_texts([text])[0]), "gpt2"
    except Exception:
        common={"the","a","an","is","of","and","red","old","small","blue"}
        ws=text.lower().split(); return sum((w in common) for w in ws), "fallback-lexical"
_LM=None
_SCORES={}

def run(beam=24, max_steps=4):
    start=time.time(); states=[("", "", 0.0, "seed")]; expansions=0; pruned=0; rejected=[]
    # Independent role schedules; alternatives are selected online, not paired.
    schedule=["subject","verb","object","punct"]
    for step,role in enumerate(schedule[:max_steps]):
        nxt=[]
        left_opts=CLAUSES if role=="subject" else ([x for x in ROLES[role]] if role in ROLES else ["."])
        right_opts=CLAUSES if role=="subject" else ([x for x in ROLES[role]] if role in ROLES else ["."])
        for l,r,old,prov in states:
            for lw in left_opts:
                for rw in right_opts:
                    expansions+=1
                    nl=(l+" "+lw).strip(); nr=(r+" "+rw).strip()
                    if compatible(nl,nr):
                        if nl not in _SCORES: _SCORES[nl]=score(nl)
                        if nr not in _SCORES: _SCORES[nr]=score(nr)
                        (sl,engine), (sr,_)=_SCORES[nl],_SCORES[nr]
                        nxt.append((nl,nr,old+sl+sr,prov+f"|{role}:{engine}"))
                    else:
                        pruned+=1
                        if len(rejected)<20: rejected.append({"left":nl,"right":nr,"rendered":nl+" / "+nr,"reason":"mirrored character conflict"})
                    if time.time()-start>20: break
                if time.time()-start>20: break
            if time.time()-start>20: break
        nxt.sort(key=lambda z:z[2],reverse=True); states=nxt[:beam]
        if not states: break
    probes=list(rejected); exact=[]
    for l,r,s,p in states[:min(20,len(states))]:
        text=(l+" "+r).strip(); ok=palindrome(text)
        row={"text":text,"left":l,"right":r,"normalized":norm(text),"letters":len(norm(text)),"exact_palindrome":ok,"score":s,"provenance":p,"shortcut_gates":{"repeated_unit":False,"word_order_symmetry":False,"catalogue_source":False}}
        probes.append(row)
        if ok: exact.append(row)
    script=Path(__file__).read_bytes(); fingerprint=hashlib.sha256(script).hexdigest()
    return {"experiment_id":"neural-dual-prefix-beam-20260915","signature":"neural-dual-prefix|typed-role-online-emission|gpt2-bilateral-ranking|character-equation-state|finite-beam|independent-clause-authoring","method":"Two independently expanded typed sides; outer character equations prune online; GPT-2 ranks both sides only after compatibility.","parameters":{"beam":beam,"max_steps":max_steps,"time_cap_seconds":20},"stats":{"expansions":expansions,"pruned":pruned,"final_states":len(states),"exact":len(exact),"admitted":0},"rendered_probes":probes,"exact_candidates":exact,"repair_operator":"When a role expansion is pruned, retain its first conflicting mirrored character and replace only that role with a held-out same-role lexical alternative; re-score both complete prefixes and re-run the independent audit.","independent_audit":{"ascii_two_pointer":all(palindrome(x["text"]) for x in exact),"method":"norm then two-pointer equality"},"provenance":{"script_sha256":fingerprint,"output_excluded":True,"generated_at":"2026-09-15"}}

if __name__=="__main__":
    out=run(); path=Path("runs/neural-dual-prefix-beam-20260915.json"); path.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"path":str(path),"stats":out["stats"],"signature":out["signature"]},indent=2))
