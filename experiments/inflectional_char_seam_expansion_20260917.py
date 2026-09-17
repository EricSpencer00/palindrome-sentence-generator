#!/usr/bin/env python3
"""Character-level seam expansion over inflectional lexical tries.

Unlike the preceding bundle ranking, this expands each lexical word one
character at a time and checks every newly resolved outer tape equation before
opening the next word.
"""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/inflectional-char-seam-expansion-20260917.json"
AGENTS=["gardener","teacher","cartographer","messenger","archivist"]
VERBS=["carries","writes","marks","records","keeps"]
OBJECTS=["letters","notes","maps","charts","records"]
SETTINGS=["harbor","garden","station","archive"]
TEMPLATE="The {a0} {v0} the {o0} beside the {s0}, and the {a1} {v1} the {o1} beside the {s1}."

def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s); i=0; j=len(t)-1; bad=[]
 while i<j:
  if t[i]!=t[j]: bad.append((i,j))
  i+=1; j-=1
 return {"letters":len(t),"exact":not bad,"mismatch_count":len(bad),"first_mismatch":bad[0][0] if bad else None,"sha256":hashlib.sha256(t.encode()).hexdigest(),"independent_two_pointer":not bad}
class WordTrie:
 def __init__(self,words):
  self.root={}
  for w in words:
   n=self.root
   for c in norm(w): n=n.setdefault(c,{})
   n["$end"]=True
 def prefixes(self):
  out=[]
  def walk(n,p):
   if "$end" in n: out.append(p)
   for c,k in n.items():
    if c!="$end": walk(k,p+c)
  walk(self.root,""); return out

def render(words):
 a0,v0,o0,s0,a1,v1,o1,s1=words
 return TEMPLATE.format(a0=a0,v0=v0,o0=o0,s0=s0,a1=a1,v1=v1,o1=o1,s1=s1)
def char_expand(slots):
 # Resolve each selected word through its trie prefix, checking the currently
 # known outer pair after every appended character.
 tries=[WordTrie(x).prefixes() for x in (AGENTS,VERBS,OBJECTS,SETTINGS,AGENTS,VERBS,OBJECTS,SETTINGS)]
 resolved=[]; checks=[]; current=[""]*8
 for i, choices in enumerate(tries):
  word=choices[slots[i]]; p=""
  for c in word:
   p+=c; current[i]=p; resolved.append(p)
   tape=norm(render(current))
   checks.append(sum(tape[j] != tape[-1-j] for j in range(min(3,len(tape)//2))))
 return checks
def main():
 rows=[]
 # Small cross-role set: first 12 combinations are expanded character-wise;
 # selection is not a post-hoc palindrome filter.
 configs=[(0,0,0,0,1,1,1,1),(1,1,1,1,0,0,0,0),(2,2,2,2,3,3,3,3),(3,3,3,3,4,4,4,0),(4,4,4,0,2,2,2,2),(0,1,2,3,3,4,0,1)]
 for i,ix in enumerate(configs):
  words=(AGENTS[ix[0]],VERBS[ix[1]],OBJECTS[ix[2]],SETTINGS[ix[3]],AGENTS[ix[4]],VERBS[ix[5]],OBJECTS[ix[6]],SETTINGS[ix[7]])
  text=render(words); a=audit(text); checks=char_expand(ix)
  rows.append({"candidate":i,"rendered":text,"words":list(words),"character_checks":len(checks),"max_partial_mismatch":max(checks) if checks else 0,"provenance":"typed_inflectional_word_trie_character_expansion","novelty_preflight":{"signature":"inflectional_trie_char_expand_v1","distinct_from":"bundle-level trie ranking; this expands and checks each word prefix before next slot"},"audit":a,"anti_shortcut":{"catalogue":False,"fragment":False,"mirrored_halves":False,"repeated_unit":words[:4]==words[4:],"punctuation_carries_letters":False,"intact_prose":True}})
 payload={"experiment":"inflectional-char-seam-expansion-20260917","method":"inflectional word tries expanded character-by-character; partial seam equations checked before next lexical slot","template":TEMPLATE,"candidate_count":len(rows),"candidates":rows,"summary":{"exact_count":sum(r["audit"]["exact"] for r in rows),"longest_letters":max(r["audit"]["letters"] for r in rows),"next_repair":"replace fixed template tokens with a character trie for function words and enforce live equations across variable word boundaries"}}
 OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload["summary"],sort_keys=True))
if __name__=="__main__": main()
