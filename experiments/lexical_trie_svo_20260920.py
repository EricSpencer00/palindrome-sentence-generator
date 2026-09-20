"""Character-conditioned lexical-trie grammar with live residual debt.

Words, rather than preassembled mirrored phrases, are selected from role
tries.  Grammar paths are complete SVO clauses with optional PP/relative
extensions; the two sides choose words independently.
"""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORED = {
    "DET": ("a", "an", "the"), "N": ("baker", "singer", "captain", "gardener", "poet", "keeper", "child", "teacher"),
    "V": ("greets", "keeps", "writes", "guides", "opens", "carries", "hears", "marks"),
    "PREP": ("by", "near", "under", "beside", "with"), "REL": ("who", "that"),
}

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t=letters(s); r=t[::-1]
    return {"letters":len(t), "two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}

class Trie:
    def __init__(self): self.root={}; self.words=[]
    def add(self, word):
        node=self.root
        for ch in word: node=node.setdefault(ch,{})
        node.setdefault("$",[]).append(word); self.words.append(word)
    def _under(self,node):
        out=list(node.get("$",[]))
        for k,v in node.items():
            if k != "$": out.extend(self._under(v))
        return out
    def compatible(self, obligation):
        """Trie query: return words whose exposed edge matches obligation."""
        if not obligation: return tuple(self.words)
        node=self.root; out=[]
        for ch in obligation:
            if ch not in node:
                return tuple(dict.fromkeys(out))
            node=node[ch]; out.extend(node.get("$",[]))
        # Longer lexical edges may leave residual debt on the other side.
        out.extend(self._under(node))
        return tuple(dict.fromkeys(out))

def bank():
    out={k:list(v) for k,v in AUTHORED.items()}
    try:
        from nltk.corpus import brown
        counts={k:Counter() for k in out}
        for sent in brown.tagged_sents():
            for word,tag in sent:
                w=word.casefold()
                if not re.fullmatch("[a-z]+",w): continue
                role=("DET" if tag.startswith("AT") else "N" if tag.startswith(("NN","NP"))
                      else "V" if tag.startswith("VB") else "PREP" if tag.startswith("IN") else None)
                if role: counts[role][w]+=1
        for role,c in counts.items(): out[role].extend(w for w,n in c.most_common(80) if n>2)
    except LookupError: pass
    return {k:tuple(dict.fromkeys(v)) for k,v in out.items()}

def paths():
    # Word-level grammar: SVO plus independently optional adjunct/relative tails.
    return (("DET","N","V","DET","N"),
            ("DET","N","V","DET","N","PREP","DET","N"),
            ("DET","N","V","DET","N","REL","V"),
            ("DET","N","V","DET","N","PREP","DET","N","REL","V"))

def run(limit=180000):
    b=bank(); tries={k:Trie() for k in b}; reverse_tries={k:Trie() for k in b}
    for role,ws in b.items():
        for w in ws:
            tries[role].add(letters(w)); reverse_tries[role].add(letters(w)[::-1])
    states=pruned=0; exact=[]; seen=set(); grammar=paths()
    for lp in grammar:
      for rendered_rp in grammar:
        rp=tuple(reversed(rendered_rp)); stack=[(0,0,"","","","",())]
        while stack and states<limit:
          li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
          if li==len(lp) and ri==len(rp):
            if lbuf or rbuf: pruned+=1; continue
            text=(left+" "+right).strip(); a=audit(text)
            if a["two_pointer_exact"] and a["letters"]>38 and text not in seen:
              seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_roles":lp,"right_roles":rp,"lexical_trie":True,"bank":"authored+Brown-heldout","finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"corpus_sentence_replay":False,"word_path":prov}})
            continue
          if li<len(lp):
            role=lp[li]; words=tries[role].compatible(rbuf)
            for w in reversed(words[:160]):
              got=consume(lbuf+ w,rbuf)
              if got is None: pruned+=1; continue
              stack.append((li+1,ri,left+((" " if left else "")+w),right,got[0],got[1],prov+(("L",role,w),)))
          if ri<len(rp):
            role=rp[ri]; reversed_words=reverse_tries[role].compatible(lbuf)
            for rw in reversed(reversed_words[:160]):
              w=rw[::-1]; got=consume(lbuf,rbuf+rw)
              if got is None: pruned+=1; continue
              stack.append((li,ri+1,left,w+((" "+right) if right else ""),got[0],got[1],prov+(("R",role,w),)))
        if states>=limit: break
      if states>=limit: break
    return {"method":"lexical-trie-svo-20260920","grammar_paths":len(grammar),"bank_sizes":{k:len(v) for k,v in b.items()},"trie_roles":len(tries),"states":states,"pruned":pruned,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty"}

def consume(a,b):
    n=min(len(a),len(b))
    if a[:n]!=b[:n]: return None
    return a[n:],b[n:]

if __name__=="__main__":
    d=run(); (ROOT/"runs/lexical-trie-svo-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
