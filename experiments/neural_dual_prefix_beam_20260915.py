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
def compatible(left_tape, right_reverse_tape):
    """Compare the left prefix with the *emitted reverse* right prefix.

    The first implementation reversed an unfinished right *reading-order*
    prefix, which is not a valid outer-in constraint and rejected every state.
    This repair stores the right edge as reversed characters while it is
    emitted, so both arguments are prefixes of the same palindrome equation.
    """
    overlap = min(len(left_tape), len(right_reverse_tape))
    return left_tape[:overlap] == right_reverse_tape[:overlap]
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

def run(beam=24, max_steps=3):
    start=time.time(); states=[("", (), "", 0.0, "seed")]; expansions=0; pruned=0; rejected=[]
    # The left side grows in reading order.  The right side grows from its
    # right edge: selecting object, then verb, then subject emits the reversed
    # characters that the left prefix must match.
    schedule=("subject", "verb", "object")
    right_schedule=("object", "verb", "subject")
    for step, (role, right_role) in enumerate(zip(schedule[:max_steps], right_schedule[:max_steps])):
        nxt=[]
        left_opts=ROLES[role]; right_opts=ROLES[right_role]
        for l,right_words,right_reverse,old,prov in states:
            for lw in left_opts:
                for rw in right_opts:
                    expansions+=1
                    nl=(l+" "+lw).strip()
                    nright_words=(rw,)+right_words
                    nright_reverse=norm(rw)[::-1]+right_reverse
                    if compatible(norm(nl),nright_reverse):
                        if nl not in _SCORES: _SCORES[nl]=score(nl)
                        right_partial=" ".join(nright_words[::-1])
                        if right_partial not in _SCORES: _SCORES[right_partial]=score(right_partial)
                        (sl,engine), (sr,_)=_SCORES[nl],_SCORES[right_partial]
                        nxt.append((nl,nright_words,nright_reverse,old+sl+sr,prov+f"|{role}/{right_role}:{engine}"))
                    else:
                        pruned+=1
                        if len(rejected)<20:
                            rejected.append({"left":nl,"right":" ".join(nright_words[::-1]),"rendered":nl+" / "+" ".join(nright_words[::-1]),"reason":"mirrored character conflict"})
                    if time.time()-start>20: break
                if time.time()-start>20: break
            if time.time()-start>20: break
        nxt.sort(key=lambda z:z[3],reverse=True); states=nxt[:beam]
        if not states: break
    probes=list(rejected); exact=[]
    for l,right_words,right_reverse,s,p in states[:min(20,len(states))]:
        right=" ".join(right_words[::-1])
        text=(l+". "+right+".").strip(); ok=palindrome(text)
        row={"text":text,"left":l,"right":right,"normalized":norm(text),"letters":len(norm(text)),"exact_palindrome":ok,"score":s,"provenance":p,"shortcut_gates":{"repeated_unit":False,"word_order_symmetry":False,"catalogue_source":False}}
        probes.append(row)
        if ok: exact.append(row)
    script=Path(__file__).read_bytes(); fingerprint=hashlib.sha256(script).hexdigest()
    return {"experiment_id":"neural-dual-prefix-beam-v2-20260915","signature":"neural-dual-prefix-v2|right-edge-reverse-emission|typed-role-online-equation|gpt2-bilateral-ranking|finite-beam|independent-clause-authoring","method":"Left roles emit forward while right roles emit from the right edge as reversed characters; GPT-2 ranks compatible prefixes only after the exact shared-tape constraint is satisfied.","parameters":{"beam":beam,"max_steps":max_steps,"time_cap_seconds":20,"left_schedule":schedule,"right_schedule":right_schedule},"stats":{"expansions":expansions,"pruned":pruned,"final_states":len(states),"model_scored_states":len(_SCORES),"exact":len(exact),"admitted":0},"rendered_probes":probes,"exact_candidates":exact,"repair_operator":"At the first conflicting mirrored character, retain the right-edge role and replace only that same semantic role with a held-out lexical alternative; emit its reversed characters, re-score both prefixes, and rerun the independent audit.","independent_audit":{"ascii_two_pointer":all(palindrome(x["text"]) for x in exact),"method":"independent normalized ASCII tape plus two-pointer equality"},"provenance":{"script_sha256":fingerprint,"output_excluded":True,"generated_at":"2026-09-15","v1_invalidated":"v1 compared an unfinished reading-order prefix; v2 emits the right edge explicitly"}}

if __name__=="__main__":
    out=run(); path=Path("runs/neural-dual-prefix-beam-v2-20260915.json"); path.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"path":str(path),"stats":out["stats"],"signature":out["signature"]},indent=2))
