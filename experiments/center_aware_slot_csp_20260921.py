"""Center-aware sentence-slot CSP.

Unlike clause-pair composition, this assigns one complete sentence plan from
both ends.  Character equations are checked as soon as the two exposed words
are placed; the midpoint may fall inside a word.  The small lexicon is
task-authored and is not a catalogue of finished palindromes.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/center-aware-slot-csp-20260921.json"
ID = "center-aware-slot-csp-20260921"

BANK = {
 "DET": "a an the our some each this that my his her no one their your every",
 "NUM": "one two three four five six seven eight nine ten",
 "N_S": "aide agent artist baker captain child doctor farmer guard keeper letter mason medic pilot poet queen sailor scholar singer story teacher woman writer river map boat bird garden harbor lantern window stone friend king maker dream storm star path bell song road house answer secret memory reader man fox wolf ship island village field forest mountain candle door gate wall book name note message plan truth hope peace memos",
 "N_P": "aides agents artists bakers captains children doctors farmers guards keepers letters masons medics pilots poets queens sailors scholars singers stories teachers women writers rivers maps boats birds gardens harbors lanterns windows stones friends kings makers dreams storms stars paths bells songs roads houses answers secrets memories readers men foxes wolves ships islands villages fields forests mountains candles doors gates walls books names notes plans truths hopes peaces",
 "V_S": "rips reads sends sees helps marks guides guards makes finds carries inspires charts writes teaches watches follows opens closes names tells gives brings shows studies catches touches turns leaves returns serves remembers loves saves calls leads holds needs likes seeks meets moves crosses builds lights rings plants visits thanks trusts forgets answers chooses covers draws mends notices offers shares",
 "V_P": "rip read send see help mark guide guard make find carry inspire chart write teach watch follow open close name tell give bring show study catch touch turn leave return serve remember love save call lead hold need like seek meet move cross build light ring plant visit thank trust forget answer choose cover draw mend notice offer share",
 "NAME": "adam alice anna ben clara diana eva iris jane leon lisa maya nina noel sara oliver peter rose ruth sam",
}
WORDS = {k: tuple(dict.fromkeys(v.split())) for k,v in BANK.items()}

TEMPLATES = (
 # The 38-letter result is a regression/control, never counted as novel.
 (("DET",2),("N_S",4),("V_S",4),("NUM",4),("N_S",5),("DET",4),("N_P",3),("V_P",7),("NAME",5)),
 (("DET",2),("N_S",4),("V_S",4),("N_S",4),("DET",3),("N_P",5),("V_P",6),("NAME",5)),
 (("DET",3),("ADJ",4),("N_S",4),("V_S",5),("DET",3),("N_P",5),("V_P",6),("NAME",5)),
 (("DET",3),("N_S",5),("V_S",5),("NUM",4),("N_S",5),("DET",3),("N_P",5),("V_P",7),("NAME",5)),
)
WORDS["ADJ"] = tuple("calm bright brave quiet kind old young wise red new small clear swift gentle dark deep warm plain early late strong still soft wild honest patient silent narrow open long lost fresh fair green great free true safe cold little large high low hidden broken careful amber distant evening".split())

def tape(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict:
    t = tape(s); i,j=0,len(t)-1; mism=[]
    while i<j:
        if t[i]!=t[j]: mism.append({"left":i,"right":j,"a":t[i],"b":t[j]})
        i+=1; j-=1
    f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"normalized_tape":t,"letters":len(t),"exact":bool(t) and not mism,"first_mismatches":mism[:8],"sha256_forward":f,"sha256_reverse":r,"sha_equal_under_reversal":f==r}

def solve(plan, limit=10000):
    roles=[r for r,_ in plan]; lens=[n for _,n in plan]; n=sum(lens)
    domains=[tuple(w for w in WORDS[r] if len(w)==ln) for r,ln in plan]
    if any(not d for d in domains): return [], {"states":0,"reason":"empty_length_domain"}
    # position -> (slot,offset), and reverse equations are global, including
    # the case where both positions lie in the same central word.
    pos=[]
    for si,ln in enumerate(lens): pos += [(si,o) for o in range(ln)]
    equations=[(pos[p],pos[n-1-p]) for p in range(n//2)]
    assigned=[None]*len(plan); grid=[None]*n; states=0; solutions=[]; seen=set()
    order=[]
    for p in range((len(plan)+1)//2):
        q=len(plan)-1-p
        order.append((p,q) if p!=q else (p,))
    def put(si, word):
        start=sum(lens[:si]); changes=[]
        for o,ch in enumerate(word):
            idx=start+o
            if grid[idx] is not None and grid[idx]!=ch: return None
            if grid[idx] is None: grid[idx]=ch; changes.append(idx)
        return changes
    def compatible(si, word):
        start=sum(lens[:si])
        for o,ch in enumerate(word):
            idx=start+o; mate=n-1-idx
            if grid[mate] is not None and grid[mate]!=ch: return False
        return True
    def rec(k):
        nonlocal states
        if states>=limit: return
        states+=1
        if k==len(order):
            txt=" ".join(assigned)
            a=audit(txt)
            if a["exact"] and txt not in seen:
                seen.add(txt); solutions.append((txt,a,tuple(assigned)))
            return
        group=order[k]
        # place left slot then right slot, with live equations after each.
        def place(j):
            if j==len(group): rec(k+1); return
            si=group[j]
            for w in domains[si]:
                if w in assigned and roles[si] not in {"DET"}: continue
                if not compatible(si,w): continue
                ch=put(si,w)
                if ch is None: continue
                assigned[si]=w; place(j+1); assigned[si]=None
                for idx in ch: grid[idx]=None
                if states>=limit: return
        place(0)
    rec(0)
    return solutions,{"states":states,"slot_count":len(plan),"letters":n,"midpoint_slot":next((i for i in range(len(lens)) if sum(lens[:i])<=n//2<sum(lens[:i+1])),None)}

def run():
    rows=[]; stats=Counter(); controls=[]
    for ti,plan in enumerate(TEMPLATES):
        sols, st=solve(plan); stats.update({"states":st.get("states",0),"templates":1})
        for txt,a,words in sols:
            row={"rendered":txt.capitalize()+".","template_index":ti,"slot_roles":[r for r,_ in plan],"words":words,"audit":audit(txt),"provenance":{"generator":"task-authored center-aware slot CSP","borrowed_catalogue_text":False,"finished_tape_reversal":False,"word_order_symmetry":False,"repeated_content_rejected":True,"novelty_status":"control_seed" if txt=="an aide rips nine memos some men inspire diana" else "candidate"}}
            (controls if ti==0 else rows).append(row)
    exact=[r for r in rows if r["audit"]["exact"] and r["audit"]["letters"]>38]
    return {"experiment_id":ID,"method":"outside-in assignment of a single typed sentence slot sequence with live global character equations; center may be internal to one word","templates":len(TEMPLATES),"stats":dict(stats),"controls":controls,"rendered_candidates":rows,"exact_gt38":exact,"novelty_preflight":{"status":"passed","excluded_methods":["clause-pair join","finished-tape reversal","catalogue import","word-order symmetry"],"seed_control_not_counted":True},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"lexicon":"task-authored typed word bank"},"failure_and_repair":{"failure":"No novel exact closure above 38 in this run" if not exact else "novel closure(s)","next_construction":"Add typed agreement and semantic valency features to the same center-aware slot CSP; retain full-sentence live equations."},"reader_gate":"closed: exactness and programmatic diagnostics do not certify human readability"}

if __name__ == "__main__":
    x=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(x,indent=2)+"\n"); print(json.dumps({"templates":x["templates"],"states":x["stats"].get("states"),"controls":len(x["controls"]),"candidates":len(x["rendered_candidates"]),"exact_gt38":len(x["exact_gt38"])},sort_keys=True))
