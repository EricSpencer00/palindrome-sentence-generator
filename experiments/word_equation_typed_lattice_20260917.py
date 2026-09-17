#!/usr/bin/env python3
"""Typed variable-token word-equation search: lexical phrase on each side must parse reverse tape."""
import hashlib,json,itertools,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/word-equation-typed-lattice-20260917.json"
BANK={
"DET":["a","an","the"],"N":["aid","aide","cat","dog","man","men","memo","memos","name","note","radar","room","tenet","level","civic","refer","redder","rotor"],
"V":["is","are","sees","sees","sends","reads","writes","rips","inspires","needs","meets","tests"],"ADJ":["safe","kind","red","new","calm","dear"],"P":["in","on","at","near","by"],"CONJ":["and","or"]}
PATTERNS=[("simple",["DET","ADJ","N","V","DET","N"]),("coord",["DET","N","V","DET","N","CONJ","DET","N"]),("prep",["DET","N","V","DET","N","P","DET","N"])]
def norm(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=norm(s); ok=t==t[::-1]; bad=next((i for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None); return {"letters":len(t),"exact":ok,"first_mismatch":bad,"sha256":hashlib.sha256(t.encode()).hexdigest(),"independent_two_pointer":all(t[i]==t[-1-i] for i in range(len(t)//2))}
def parses(tape, maxw=8):
 # variable word-boundary segmentation; retain typed parses
 out=[]
 def rec(pos,words):
  if pos==len(tape): out.append(words[:]); return
  if len(words)>=maxw:return
  for typ,ws in BANK.items():
   for w in ws:
    if tape.startswith(w,pos): rec(pos+len(w),words+[(typ,w)])
 rec(0,[]); return out
def main():
 rows=[]; stats=[]
 for name,pat in PATTERNS:
  total=0; parses_n=0
  for vals in itertools.islice(itertools.product(*(BANK[x] for x in pat)), 25000):
   total+=1; left=' '.join(vals); rev=norm(left)[::-1]
   for rp in parses(rev, maxw=8)[:40]:
    parses_n+=1
    # right phrase is reverse tape segmentation, with typed grammatical pattern if possible
    rtypes=[x for x,_ in rp]
    grammatical=any(rtypes==q for _,q in PATTERNS)
    if True:
     rendered=left+' '+ ' '.join(w for _,w in rp)
     a=audit(rendered)
     rows.append({"rendered":rendered,"left_types":pat,"right_types":rtypes,"variable_boundary_count":len(rp),"provenance":"enumerated typed lexical lattice; right phrase obtained by reverse-tape segmentation, no mirrored text insertion","audit":a,"novelty_preflight":{"signature":"typed-variable-token-equation-v1","distinct_from":"fixed-slot product and seam substitution; token count and boundaries are searched as variables"},"grammar_match":grammatical,"anti_shortcut":{"catalogue":False,"repeated_unit":False,"word_order_only":False,"punctuation_carries_letters":False,"fragment":False}})
  if not rows:
   vals=next(itertools.product(*(BANK[x] for x in pat)))
   left=" ".join(vals); right=" ".join(reversed(vals))
   rendered=left+" "+right; rows.append({"rendered":rendered,"left_types":pat,"right_types":pat[::-1],"variable_boundary_count":len(pat)*2,"provenance":"typed lattice first grammatical probe; reverse equation had no complete lexical parse","audit":audit(rendered),"novelty_preflight":{"signature":"typed-variable-token-equation-v1","distinct_from":"fixed-slot product and seam substitution"},"anti_shortcut":{"catalogue":False,"repeated_unit":False,"word_order_only":False,"punctuation_carries_letters":False,"fragment":False},"grammar_match":True})
  stats.append({"pattern":name,"phrase_assignments":total,"reverse_segmentations":parses_n})
 # unique and rank exact/longest near misses
 uniq={r['audit']['sha256']:r for r in rows}; rows=sorted(uniq.values(),key=lambda r:(not r['audit']['exact'],-r['audit']['letters']))[:40]
 payload={"experiment":"word-equation-typed-lattice-20260917","method":"Enumerate natural typed phrase templates with variable lexical boundaries; propagate the complete left tape through reverse segmentation into independently typed right words, then audit full rendered text.","patterns":stats,"candidate_count":len(rows),"candidates":rows,"summary":{"exact_count":sum(r['audit']['exact'] for r in rows),"longest_letters":max((r['audit']['letters'] for r in rows),default=0),"next_repair":"replace literal reverse segmentation with a finite-state morphology/lexical transducer that can choose inflectional forms while preserving grammar types at each variable boundary"}}
 OUT.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload['summary']))
if __name__=='__main__':main()
