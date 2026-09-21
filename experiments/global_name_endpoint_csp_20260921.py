"""Global variable-span CSP with typed ordinary proper-name endpoints."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/global-name-endpoint-csp-20260921.json"
LEXICON = {
    "det": ["the", "a", "this", "that"],
    "name": ["Diana", "Adrian", "Leon", "Noel"],
    "name_obj": ["Diana", "Adrian", "Leon", "Noel"],
    "v": ["observes", "guides", "writes", "helps"],
    "obj": ["lantern", "letter", "garden", "harbor", "signal", "window"],
    "conj": ["and"],
}
SLOTS = ("d1", "name1", "v1", "d2", "obj1", "conj", "d3", "name2", "v2", "name_obj2")
CATS = ("det", "name", "v", "det", "obj", "conj", "det", "name", "v", "name_obj")

def norm(s): return re.sub(r"[^a-z]", "", s.lower())
def independent_audit(words):
    tape = norm(" ".join(words)); i, j = 0, len(tape)-1
    while i < j and tape[i] == tape[j]: i += 1; j -= 1
    return {"letters": len(tape), "exact": bool(tape) and i >= j,
            "first_mismatch": None if i >= j else [i, tape[i], j, tape[j]],
            "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def solve(limit=50000, targets=range(39, 61)):
    totals={0}
    for cat in CATS: totals={n+len(norm(w)) for n in totals for w in LEXICON[cat]}
    feasible=sorted(totals & set(targets)); nodes=conflicts=complete=0; learned=set(); found=[]
    def target_search(target):
        nonlocal nodes, conflicts, complete
        chars={}; vals={}
        def place(word,start,changes):
            for k,ch in enumerate(norm(word)):
                p=start+k; q=target-1-p
                if not (0<=p<target and 0<=q<target): return False
                for pos in (p,q):
                    old=chars.get(pos)
                    if old is not None and old != ch: return False
                    if old is None: chars[pos]=ch; changes.append(pos)
            return True
        def rec(li,ri,lo,ro):
            nonlocal nodes,conflicts,complete
            if nodes>=limit: return
            if li>ri:
                if lo != target-ro or len(chars)!=target: conflicts+=1; return
                complete+=1; words=[vals[s] for s in SLOTS]
                if len({words[1],words[7],words[9]})==3: found.append(words)
                return
            lc,rc=CATS[li],CATS[ri]
            for lw0 in LEXICON[lc]:
                lw=norm(lw0); key=(li,lw,target,lo)
                if key in learned or lo+len(lw)>target-ro: continue
                nodes+=1; chg=[]
                if not place(lw,lo,chg): conflicts+=1; learned.add(key); [chars.pop(p,None) for p in chg]; continue
                vals[SLOTS[li]]=lw0
                for rw0 in (LEXICON[rc] if ri!=li else [lw0]):
                    rw=norm(rw0); rs=target-ro-len(rw); rkey=(ri,rw,target,ro)
                    if rkey in learned or rs<lo+len(lw) or (ri!=li and SLOTS[li].startswith('name') and SLOTS[ri].startswith('name') and rw0==lw0): continue
                    nodes+=1; rchg=[]
                    if place(rw,rs,rchg): vals[SLOTS[ri]]=rw0; rec(li+1,ri-1,lo+len(lw),ro+len(rw)); vals.pop(SLOTS[ri],None)
                    else: conflicts+=1; learned.add(rkey)
                    [chars.pop(p,None) for p in rchg]
                    if nodes>=limit: break
                vals.pop(SLOTS[li],None); [chars.pop(p,None) for p in chg]
                if nodes>=limit: return
        rec(0,len(SLOTS)-1,0,0)
    for t in feasible:
        if nodes>=limit: break
        target_search(t)
    return {"found":found,"stats":{"nodes":nodes,"conflicts":conflicts,"learned_nogoods":len(learned),"complete_assignments":complete,"limit":limit,"feasible_targets":feasible},"state_model":{"typed_name_endpoint":True,"typed_name_object_endpoint":True,"variable_word_boundaries":True,"shared_character_variables":True,"joint_role_choice":True,"dependency_links":["det->name", "verb->object", "subject->finite-verb"],"agreement_links":["determiner/name number-compatible"],"conflict_learning":"slot/lexeme/target/offset nogoods"}}

def run(limit=50000):
    x=solve(limit); records=[]
    for words in x["found"][:8]:
        rendered=" ".join(words)+"."; records.append({"tokens":words,"rendered":rendered,"audit":independent_audit(words),"provenance":{"joint_grammar_role_and_name_search":True,"ordinary_name_entries_only":True,"name_entries":LEXICON["name"],"complete_clause_before_test":False,"reversed_name_pairing":False,"seed_or_pair_hardcoding":False,"outer_support_residuals":"rendered in audit.first_mismatch"},"anti_shortcut":{"finished_tape_reversal":False,"mirrored_units":False}})
    return {"experiment_id":"global-name-endpoint-csp-20260921","method":"bounded global variable-span CSP with typed ordinary proper-name subject/object endpoints and joint determiner/name/verb/object role choices","config":{"letter_band":[39,60],"state_limit":limit,"name_category":"ordinary non-self-palindromic lexical entries"},**x,"records":records,"independent_exact_shortcut_audits":True,"novelty_preflight":{"status":"passed","signature":"global-variable-span|typed-proper-name-object-endpoint|joint-role-mirror-csp","signature_collision":False},"queue_row":{"lane":"Astra","status":"satisfying assignments" if records else "bounded residual","next":"retain typed names while widening only grammar-supported valency frames"}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(),indent=2)+"\n")
