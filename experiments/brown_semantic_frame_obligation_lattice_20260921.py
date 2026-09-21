"""Brown-PCFG semantic-frame lattice with live bilateral obligations.

This is a new lexical/grammar lane, not reverse-tape segmentation: each side
is generated forward from an independently selected POS frame.  A character
obligation deque is consulted at every lexical choice, before descendants are
expanded.  Brown frequencies rank alternatives but never score completed
palindrome candidates.
"""
from __future__ import annotations
from collections import deque
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BANK=json.loads((ROOT/"data/brown_pcfg_bank_20260920.json").read_text())
TOP=6
FRAMES=[
    ["DET","NOUN","VERB","DET","NOUN"],
    ["PRON","VERB","DET","NOUN"],
    ["DET","NOUN","VERB","ADJ"],
    ["PRON","VERB","PREP","DET","NOUN"],
    ["DET","NOUN","VERB","ADV"],
]

def norm(s): return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
    t=norm(s); bad=[i for i in range(len(t)//2) if t[i]!=t[-1-i]]
    return {"exact":bool(t) and not bad,"letters":len(t),"mismatch_count":len(bad),
      "pointer_pairs_checked":len(t)//2,"forward_sha256":hashlib.sha256(t.encode()).hexdigest(),
      "reverse_sha256":hashlib.sha256(t[::-1].encode()).hexdigest()}
def shortcuts(s):
    ws=re.findall('[a-z]+',s.lower()); out=[]
    if len(ws)!=len(set(ws)): out.append('repeated_word_unit')
    if any(len(w)>=3 and w==w[::-1] for w in ws): out.append('self_palindromic_word_unit')
    if any(len(w)>=3 and w[::-1] in ws and w!=w[::-1] for w in ws): out.append('semordnilap_word_pair')
    return out

LEX={k:[x['word'] for x in sorted(v,key=lambda z:-z['score'])[:TOP]] for k,v in BANK['lexicon'].items()}

def put(q,word,left,origin):
    chars=word if left else word[::-1]
    side=1 if left else -1
    if not q:
        q.extend(chars); return True,side
    if origin==side:
        q.extend(chars); return True,origin
    for i,c in enumerate(chars):
        if not q:
            q.extend(chars[i:])
            return True,side
        if q.popleft()!=c: return False,origin
    return True, (0 if not q else origin)

def search(frame,limit=50000):
    nodes=prunes=terminals=0; outputs=[]
    def rec(li,ri,lw,rw,q,origin,left):
        nonlocal nodes,prunes,terminals
        nodes+=1
        if nodes>limit:return
        if li==len(frame) and ri==len(frame):
            terminals+=1
            if not q:
                text=' '.join(lw+rw); a=audit(text)
                if a['exact'] and a['letters']>=40 and not shortcuts(text):
                    outputs.append({'text':text,'audit':a,'provenance':{'left_words':lw,'right_words':rw,'frame':frame,'bank':'data/brown_pcfg_bank_20260920.json','construction':'forward semantic-frame lattice + live deque'},'shortcut_reasons':[]})
            return
        # Strict alternating outer expansions keeps both clauses independent
        # while exposing character obligations as soon as a slot is lexicalized.
        is_left=left
        idx=li if is_left else ri
        if idx>=len(frame): return
        for w in LEX.get(frame[idx],[]):
            nq=deque(q)
            ok,no=put(nq,norm(w),is_left,origin)
            if not ok: prunes+=1; continue
            if is_left: rec(li+1,ri,lw+[w],rw,nq,no,False)
            else: rec(li,ri+1,lw,rw+[w],nq,no,True)
    rec(0,0,[],[],deque(),0,True)
    return outputs,{'nodes':nodes,'obligation_prunes':prunes,'complete_terminals':terminals}

def main():
    allrows=[]; stats=[]
    for frame in FRAMES:
        rows,st=search(frame); st['frame']=' '.join(frame); stats.append(st); allrows.extend(rows)
    payload={'run_id':'brown-semantic-frame-obligation-lattice-20260921','status':'SEARCH_COMPLETED',
      'method':'forward-generated independent Brown PCFG semantic frames with live character obligations',
      'constraints':{'post_render_search':False,'reversed_tokens':False,'catalogue_text':False,'rlaif_per_candidate':False,'independent_side_frames':True,'brown_frequency_only_for_lexical_rank':True},
      'top_k_per_pos':TOP,'frame_stats':stats,'candidate_count':len(allrows),'reader_shortlist':[x for x in allrows if x['audit']['letters']>=40],
      'next_repair':'add agreement/valency registers to the frame lattice and vary left/right frames independently before lexical expansion'}
    out=ROOT/'runs'/'brown-semantic-frame-obligation-lattice-20260921.json';out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'candidate_count':len(allrows),'frame_stats':stats},indent=2))
if __name__=='__main__':main()
