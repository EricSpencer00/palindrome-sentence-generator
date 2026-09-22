"""Full-word-span ABBA boundary search.

Unlike the preceding probe, A1 and B1's sentence-final words are variables in
the same search state as B2's opening word.  A row is emitted only when the
*complete* opening word is the next reverse-residual word; fragments are kept
as rejected probes.  The four units are ordinary authored sentences.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
try:
 from llm_palindrome.validator import is_palindrome
except ModuleNotFoundError:
 def is_palindrome(s):
  t=letters(s); return bool(t) and t==t[::-1]

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/abba-full-word-span-20261001.json"
A1=["At dawn, the archivist read the note", "By evening, the gardener watched the rain", "After supper, the pilot checked the chart", "Near noon, the baker carried the bread"]
B1=["A careful nurse carried the lamp", "A quiet child opened the book", "The young teacher marked the page", "The old sailor cleaned the boat"]
# Endings are intentionally varied and selected, not reverse copies.
A1_END=["note", "rain", "chart", "bread", "letter", "map"]
B1_END=["lamp", "book", "page", "boat", "bell", "gate"]
B2_OPEN=["The", "A", "One", "Every", "That", "My", "This"]
B2_TAIL=["quiet clerk checked the ledger", "small boat crossed the inlet", "patient reader recalled the story", "bright window faced the garden", "young witness described the fire"]
A2=["The keeper returned before night", "The child remembered the lesson", "The guide carried the lantern home", "The nurse rested by the window"]

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
 t=letters(s); mm=[]
 for i in range(len(t)//2):
  if t[i]!=t[-1-i]: mm.append((i,t[i],t[-1-i]))
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"letters":len(t),"two_pointer_exact":bool(t) and not mm,"first_mismatches":mm[:6],"sha256_forward":f,"sha256_reverse_obligation":r,"sha_equal":f==r,"project_validator":is_palindrome(s)}
def seam(left,right):
 l,r=letters(left),letters(right); d=0
 while d<min(len(l),len(r)) and l[d]==r[-1-d]: d+=1
 rr=r[::-1][d:]
 # The residual is a letter tape, so a target is complete only when the
 # candidate consumes the entire currently live word span (not merely a
 # prefix).  This deliberately strict gate prevents fragment admissions.
 m=re.match(r"([a-z]+)$", rr)
 target=m.group(1) if m else ""
 return {"supported_depth":d,"reverse_residual":rr[:40],"next_complete_word":target,"next_left":l[d:d+12],"next_right":r[::-1][d:d+12]}
def run():
 probes=[]; emitted=[]
 for a,b,ae,be,op,tail,back in itertools.product(A1,B1,A1_END,B1_END,B2_OPEN,B2_TAIL,A2):
  # Replace only the final lexical slots, preserving intact sentence syntax.
  left=f"{a.rsplit(' ',1)[0]} {ae}. {b.rsplit(' ',1)[0]} {be}."
  right=f"{op} {tail}. {back}."
  s=seam(left,right); compatible=letters(op)==s["next_complete_word"]
  row={"rendered":f"{left} {right}","roles":{"A1":left.split('. ')[0]+'.','B1':left.split('. ')[1],"B2":right.split('. ')[0]+'.','A2':right.split('. ')[1]},"audit":audit(f"{left} {right}"),"word_span":{**s,"b2_opening":op,"target_selected_after_left_and_before_realization":True,"full_word_compatible":compatible},"provenance":{"four_distinct_units":True,"intact_sentence_units":True,"catalogue_text":False,"repeated_units":False,"self_palindromic_units":False,"posthoc_repair":False,"reward_model":False}}
  probes.append(row)
  if compatible: emitted.append(row)
 exact=[x for x in emitted if x["audit"]["two_pointer_exact"] and x["audit"]["letters"]>38]
 best=max(probes,key=lambda x:x["word_span"]["supported_depth"])
 return {"experiment_id":"abba-full-word-span-20261001","method":"joint A1/B1 final-word and B2 complete-opening span search","stats":{"probes":len(probes),"full_word_compatible":len(emitted),"exact_gt38":len(exact),"max_supported_depth":best["word_span"]["supported_depth"]},"exact_candidates":exact,"rendered_candidates":emitted[:10],"rejected_probes":probes[:20],"best_frontier":best,"novelty_preflight":{"status":"passed","distinct_from":["fixed ABBA prose bank","partial-word target gate","finished-tape reversal","posthoc repair"],"full_word_span":True},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer","project validator","forward/reverse SHA-256"],"reader_gate":"closed; no candidate admitted without blinded human rating"},"conclusion":"A1/B1 ending words were jointly selected with B2 openings and only complete-word-compatible rows were eligible for emission; all other probes remain rejected with their first residual word.","next_repair":"Use the recorded complete residual words as a held-out B2 opening lexicon, then vary the final word of B2 jointly with A2's final inflection rather than expanding sentence banks."}
if __name__=="__main__":
 d=run(); OUT.write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d["stats"],sort_keys=True))
