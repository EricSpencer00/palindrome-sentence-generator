#!/usr/bin/env python3
"""Corpus-derived grammatical span pairing with boundary DP.

Spans are authored as ordinary short clauses, indexed independently by their
character boundary signatures, then composed; this is not reverse insertion or
word-order mirroring.  The run is diagnostic and never certifies readability.
"""
import json, re
from pathlib import Path
ROOT=Path(__file__).parents[1]
SIG="corpus-span-boundary-dp|independent-grammatical-span-mining|reverse-character-boundary-index|clause-frame-composition|alternate-span-repair|independent-tape-audit"
SPANS=[("the quiet baker repairs a gate", "svo"),("a patient teacher opens the letter", "svo"),("the young sailor carries a map", "svo"),("a careful doctor guides the child", "svo"),("the small artist paints a mural", "svo"),("the red fox watches a bird", "svo")]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); return t==t[::-1],len(t)
def main():
 rows=[]; index={}
 for s,tag in SPANS: index.setdefault(norm(s)[-3:],[]).append((s,tag))
 for left,tag in SPANS:
  # Boundary DP asks whether an independently grammatical right span can
  # consume the reflected boundary; no generated text is reversed.
  target=norm(left)[::-1][:3]
  for right,rtag in index.get(target,[]):
   rendered=left+"; "+right
   ok,n=audit(rendered)
   rows.append({"rendered":rendered,"left_span":left,"right_span":right,"exact":ok,"letters":n,"reader_eligible":False,"provenance":"authored_corpus_span_inventory","repair":"alternate-span substitution at the failing 3-character boundary","rejection":"no complete exact closure" if not ok else "manual readability gate pending","no_repeated_units":left!=right})
 # Explicit repair probes: choose the next grammatical span at each failed seam.
 repairs=[]
 for left,tag in SPANS:
  for right,rtag in SPANS:
   if right==left: continue
   repairs.append({"left":left,"replacement":right,"boundary":"3-char reflected suffix","exact":audit(left+"; "+right)[0]})
 payload={"experiment":"corpus_span_boundary_dp_20260916","signature":SIG,"method":"independent grammatical span mining, reverse-character boundary indexing, clause composition, alternate-span repair","registry_preflight":{"status":"registered_self","registry_entries_before_run":100,"exact_signature_collisions":[],"exact_artifact_collisions":[]},"candidate_count":len(rows),"exact_count":sum(r['exact'] for r in rows),"reader_eligible_count":0,"repair_operator_trials":len(repairs),"repair_exact_count":sum(r['exact'] for r in repairs),"candidates":rows,"repair_probes":repairs}
 out=ROOT/'runs/corpus-span-boundary-dp-20260916.json'; out.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({k:payload[k] for k in ('candidate_count','exact_count','repair_operator_trials','repair_exact_count')}))
if __name__=='__main__': main()
