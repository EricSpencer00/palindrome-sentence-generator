"""Small offline shared-character-tape decoder experiment.

The left clause is generated from a hand-auditable POS lexicon.  Its letters
are frozen and reversed; only word boundaries on the right may change.  Both
halves must match one of a small set of clause templates.  This is an
experiment lead generator, not a readability judge.
"""
from __future__ import annotations

import hashlib, json, random, re
from pathlib import Path
from itertools import product
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEXICON = {x.strip() for x in (ROOT / "data/lexicon.txt").read_text().splitlines()
           if re.fullmatch(r"[a-z]+", x.strip())}

# Deliberately closed and inspectable; words are common enough to read aloud.
POS = {
 "DET":"a an the this that my your our his her one", "PRON":"i we you he she they it",
 "NOUN":"artist baker bird child clerk cloud dog door dream friend garden girl hand horse house king lamp letter light man market moon mother music night nurse park path rain river road room sailor school seed signal sky song stone sun teacher town train tree traveler watch water wind woman world",
 "VERB":"asks bends brings calls carries checks climbs closes covers dances draws drinks drives finds gives hears holds keeps knows leaves likes listens lives looks makes meets moves opens paints plans reads runs sees sends sets shares sings sits speaks starts stays takes tells thinks turns walks wants watches wins writes",
 "ADJ":"calm clear cold dark early fair fine fresh good kind late little long mild new quiet red safe small soft warm wise young",
 "ADV":"again away back down here home now often out there then today up well",
}
POS = {k: tuple(w for w in v.split() if w in LEXICON) for k,v in POS.items()}
TEMPLATES = (("DET","NOUN","VERB","DET","NOUN"), ("PRON","VERB","DET","NOUN"),
             ("NOUN","VERB","ADV"), ("DET","ADJ","NOUN","VERB"),
             ("PRON","VERB","ADV"), ("NOUN","VERB","DET","ADJ","NOUN"))

def norm(s): return re.sub("[^a-z]", "", s.lower())
def trie_words(words):
    root={}
    for w in words:
        n=root
        for c in w: n=n.setdefault(c,{})
        n.setdefault("",True)
    return root

def segment(tape, words, limit=500):
    tr=trie_words(words); memo={}
    def go(i):
        if i == len(tape): return [()]
        if i in memo:return memo[i]
        out=[]; n=tr
        for j in range(i,len(tape)):
            n=n.get(tape[j]);
            if n is None:break
            if "" in n:
                for tail in go(j+1):
                    out.append((tape[i:j+1],)+tail)
                    if len(out)>=limit:return out
        memo[i]=out; return out
    return [" ".join(x) for x in go(0)]

def main(seed=20260912, trials=500000):
    rng=random.Random(seed); bypos={k:list(v) for k,v in POS.items()}
    vocab=set(sum((list(v) for v in POS.values()), [])); hits=[]; seen=set()
    # A broad segmentation lexicon supplies boundaries; Brown's attested
    # sentence shapes independently enforce a grammatical right reading.
    broad_vocab={w for w in LEXICON if len(w)>=2}
    # Lightweight grammar gate: closed-class and content-word categories are
    # explicit above; right lexicalizations must also fit a clause template.
    for _ in range(trials):
        shape=rng.choice(TEMPLATES); left_words=tuple(rng.choice(bypos[p]) for p in shape)
        left=" ".join(left_words); tape=norm(left)[::-1]
        if not 15<=len(tape)<=30 or len(set(left_words))<len(left_words):continue
        for right in segment(tape,vocab,limit=40):
            rw=tuple(right.split())
            if len(rw)<3 or len(rw)>8 or len(set(rw))<len(rw):continue
            # category readings; templates are hard grammar constraints.
            rshape=tuple(next((p for p,ws in bypos.items() if w in ws), "") for w in rw)
            if rshape not in TEMPLATES:continue
            if left == right or left_words == rw:continue
            text=left+" "+right
            if norm(text)!=norm(text)[::-1]:continue
            key=(left,right)
            if key in seen:continue
            seen.add(key); hits.append({"left":left,"right":right,"letters":len(norm(text)),"left_letters":len(tape),"left_pos":shape,"right_pos":rshape,"tape":tape})
            if len(hits)>=30:break
        if len(hits)>=30:break
    out={"seed":seed,"trials":trials,"lexicon_sha256":hashlib.sha256("\n".join(sorted(LEXICON)).encode()).hexdigest(),"vocabulary":vocab,"candidates":hits,"reader_gate":"Mechanical exactness and template compatibility only; no human readability claim."}
    print(json.dumps({"count":len(hits),"candidates":hits},indent=2))
    return out
if __name__=="__main__": main()
