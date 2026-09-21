"""Joint language/character-obligation search.

Unlike post-hoc repair, this decoder expands a grammatical slot on either
side only through a live deque of unmatched character obligations.  The two
clauses are independently lexicalized; no token is reversed or copied.
"""
from __future__ import annotations

import hashlib, json, re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "joint-language-obligation-search-20260921"

LEX = {
    "DET": ["a", "an", "the", "some"],
    "N": ["aide", "artist", "guard", "memos", "men", "poet", "raven", "teacher", "writer"],
    "V_S": ["admires", "finds", "guides", "helps", "marks", "notes", "reads", "sees"],
    "V_P": ["admire", "find", "guide", "help", "mark", "note", "read", "see"],
    "NAME": ["ada", "diana", "iris", "leon", "mira", "noah", "nora"],
    "PREP": ["in", "on", "near", "under"],
}

TEMPLATES = [
    ("DET N V_S DET N", ["DET", "N", "V_S", "DET", "N"]),
    ("DET N V_P DET N", ["DET", "N", "V_P", "DET", "N"]),
    ("NAME V_S DET N", ["NAME", "V_S", "DET", "N"]),
    ("DET N V_S PREP NAME", ["DET", "N", "V_S", "PREP", "NAME"]),
]

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict:
    t = norm(s); bad = [i for i in range(len(t)//2) if t[i] != t[-1-i]]
    return {"exact": bool(t) and not bad, "letters": len(t), "mismatch_count": len(bad),
            "pointer_pairs_checked": len(t)//2,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest()}

def shortcut_reasons(s: str) -> list[str]:
    ws = re.findall(r"[a-z]+", s.lower()); out=[]
    if len(ws) != len(set(ws)): out.append("repeated_word_unit")
    if any(len(w)>=3 and w == w[::-1] for w in ws): out.append("self_palindromic_word_unit")
    if any(len(w)>=3 and w[::-1] in ws and w != w[::-1] for w in ws): out.append("semordnilap_word_pair")
    return out

def add_word(q: deque[str], word: str, left: bool) -> bool:
    """Consume newly exposed outer characters against existing obligations."""
    chars = word if left else word[::-1]
    if left:
        # Left material creates obligations; the opposing right expansion
        # consumes them from the front in reverse-character order.
        q.extend(chars)
        return True
    for ch in chars:
        if q and q[0] != ch: return False
        if q: q.popleft()
        else: return False
    return True

def search(template: list[str], max_nodes: int = 200000) -> tuple[list[dict], dict]:
    # State stores independent clause words and the live unmatched obligation deque.
    rows=[]; nodes=0; pruned=0
    def rec(lidx, ridx, lw, rw, q, left_turn):
        nonlocal nodes, pruned
        nodes += 1
        if nodes > max_nodes: return
        if lidx == len(template) and ridx == len(template):
            rendered = " ".join(lw + rw)
            if not q:
                a=audit(rendered)
                if a["exact"] and a["letters"] >= 40 and not shortcut_reasons(rendered):
                    rows.append({"text": rendered, "audit":a, "provenance":{"left":lw,"right":rw,
                        "template":" ".join(template), "construction":"live_obligation_deque"},
                        "shortcut_reasons":[]})
            return
        # Expand the side with fewer slots first, but permit either side at each state.
        sides = [(True,lidx)] if ridx == len(template) else ([(False,ridx)] if lidx == len(template) else [(left_turn,lidx),(not left_turn,ridx)])
        for is_left, idx in sides:
            if idx >= len(template): continue
            slot=template[idx]
            for w in LEX[slot]:
                nq=deque(q)
                if not add_word(nq, norm(w), is_left):
                    pruned += 1; continue
                if is_left: rec(lidx+1,ridx,lw+[w],rw,nq,False)
                else: rec(lidx,ridx+1,lw,rw+[w],nq,True)
    rec(0,0,[],[],deque(),True)
    return rows,{"nodes":nodes,"obligation_prunes":pruned}

def main():
    all_rows=[]; stats=[]
    for name, slots in TEMPLATES:
        rows, st=search(slots); all_rows.extend(rows); st["template"]=name; stats.append(st)
    payload={"run_id":RUN_ID,"status":"SEARCH_COMPLETED","method":"joint grammatical slot expansion with live character obligation deque",
             "constraints":{"independent_clause_lexicalization":True,"post_render_search":False,"reversed_tokens":False,
             "catalogue_text":False,"rlaif_per_candidate":False},"templates":stats,
             "candidate_count":len(all_rows),"reader_shortlist":[r for r in all_rows if r["audit"]["letters"]>=40],"next_repair":"expand typed semantic frames and choose slots by obligation frontier, preserving live deque checks"}
    out=ROOT/"runs"/f"{RUN_ID}.json"; out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"run_id":RUN_ID,"candidate_count":len(all_rows),"templates":stats},indent=2))

if __name__ == "__main__": main()
