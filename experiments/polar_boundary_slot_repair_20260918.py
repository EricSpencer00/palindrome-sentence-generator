"""Bounded semantic repair around the 44-letter polar-question boundary.

It varies typed slots in two intact question/answer-like clauses.  Candidate
pairs containing a reversible word pair or any proper multiword palindrome are
discarded before admission; the 44-letter diagnostic is retained only as an
excluded control.  This is a new local lane, not the 710-clause Cartesian run.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

@dataclass(frozen=True)
class Frame:
    subject:str; noun:str; adjunct:str; tail:str; source:str
    @property
    def text(self): return f"{self.subject} {self.noun}{self.adjunct}{self.tail}"

SUBJECTS=("Was Noel","Was Leon","Was Diana","Can Noel","Did Leon")
NOUNS=("an era","a gas","an item","a poem","a note","the map")
ADJUNCTS=(""," in town"," at dawn"," near home")
TAILS=("?","? Met in a")
RIGHT=("saga","arena","a poem","a note","the map")
ENDINGS=("Leon saw.","Noel saw.","Diana saw.","Leon smiled.")
CONTROL="Was Noel an era, a gas, an item? Met in a, saga, arena, Leon saw."

def tape(s): return normalize_letters(s)
def audit(s):
 t=tape(s); bad=[(i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {"letters":len(t),"two_pointer_exact":not bad,"mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"sha_equal_under_reversal":hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(t[::-1].encode()).hexdigest()}

def forbidden_hidden(s):
 words=re.findall(r"[A-Za-z]+",s.lower()); bad=[]
 for i in range(len(words)):
  for j in range(i+1,len(words)):
   span=tape(" ".join(words[i:j+1]))
   if len(span)>=8 and span==span[::-1]: bad.append({"start":i,"end":j,"text":" ".join(words[i:j+1])})
 for w in words:
  if len(w)>=4 and w==w[::-1]: bad.append({"word":w})
 return bad

def run(max_probes=5000):
 rows=[]; exact=[]; checked=0
 frames=[Frame(s,n,a,t,"polar_boundary_slot_bank_v1") for s in SUBJECTS for n in NOUNS for a in ADJUNCTS for t in TAILS]
 for l in frames:
  for rword in RIGHT:
   for ending in ENDINGS:
    checked+=1; rendered=f"{l.text}, {rword}, {ending}"
    hidden=forbidden_hidden(rendered); a=audit(rendered); checks=mechanical_admission_checks(rendered,min_letters=30,max_letters=240)
    row={"rendered":rendered,"left_frame":l.__dict__,"right_slots":{"middle":rword,"ending":ending},"audit":a,"forbidden_hidden_spans":hidden,"mechanical_checks":checks,"independent_exact":a["two_pointer_exact"],"mechanically_admitted":a["two_pointer_exact"] and not hidden and all(checks.values()),"novelty":"excluded_control" if tape(rendered)==tape(CONTROL) else "new_tape","reader_status":"not_run; programmatic checks diagnose only"}
    if a["two_pointer_exact"]: exact.append(row)
    if len(rows)<max_probes: rows.append(row)
    if checked>=max_probes: break
   if checked>=max_probes: break
  if checked>=max_probes: break
 admitted=[x for x in exact if x["mechanically_admitted"]]
 return {"status":"polar_boundary_slot_repair_bounded","experiment_id":"polar-boundary-slot-repair-20260918","signature":"polar-question|typed-slot-substitution|hidden-span-rejection|independent-audit","config":{"frame_count":len(frames),"max_probes":max_probes},"stats":{"frames":len(frames),"pairs_checked":checked,"bounded":checked<len(frames)*len(RIGHT)*len(ENDINGS),"stored_probes":len(rows),"exact":len(exact),"mechanically_admitted":len(admitted),"reader_eligible":0},"control":{"rendered":CONTROL,"audit":audit(CONTROL),"status":"diagnostic only; rejected if hidden span present"},"rendered_candidates_and_probes":rows,"exact_candidates":exact,"admitted":admitted,"provenance":{"authored_polar_frames":True,"direct_reversible_word_pairs_forbidden":True,"hidden_palindromic_spans_forbidden":True,"catalogue_text_imported":False,"independent_validator":"two-pointer normalized tape plus forward/reverse SHA-256","human_readability_certified":False},"next_repair":"retain polar question frames but solve character residual at the noun/adjunct seam with held-out verbs, without permitting any reversible word or hidden span","reader_gate":"closed until an exact candidate passes intact-prose review and blinded controls"}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); ap.add_argument('--max-probes',type=int,default=5000); a=ap.parse_args(); r=run(a.max_probes); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],indent=2))
if __name__=='__main__': main()
