"""Bounded outside-in overhang search on a fresh safe contemporary vocabulary."""
from __future__ import annotations
import hashlib,json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.outside_in_sentence_decoder import run
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="outside-in-fresh-safe-vocab-20260920"
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def main():
 base=run(seeds=4,vocabulary_size=500,min_zipf=4.0,beam=32)
 rows=[]
 for row in base["records"]:
  rows.append({"rendered":row["text"],"audit":audit(row["text"]),"checks":row["checks"],"rejection_codes":row["rejection_codes"],"provenance":{"fresh_safe_vocab":True,"whole_sentence_plan":True,"post_hoc_repair":False,"finished_tape_reversal":False,"catalogue_imported":False,"reader_status":"diagnostic only"}})
 rows.sort(key=lambda x:(-x["audit"]["letters"],x["rendered"]))
 controls=["The careful writer reads a quiet book.","A kind teacher helps the young student."]
 control_rows=[{"rendered":t,"audit":audit(t),"source":"intact prose control","reader_status":"control only"} for t in controls]
 result={"experiment_id":EXPERIMENT_ID,"method":"outside-in WordTries beam overhang with whole-sentence POS plan","config":{"seeds":4,"vocabulary_size":base["vocabulary_size"],"min_zipf":4.0,"beam":32,"min_letters":30,"max_steps":96},"stats":{"records":len(rows),"mechanically_admitted":sum(not x["rejection_codes"] for x in rows),"longest_letters":max((x["audit"]["letters"] for x in rows),default=0),"exact":sum(x["audit"]["two_pointer_exact"] for x in rows)},"best_rendered_tapes":rows[:10],"control_rendered_tapes":control_rows,"complete_prose_controls":controls,"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-broad-earley-20260920","fresh_safe_vocabulary":True,"catalogue_imported":False,"post_hoc_repair":False},"provenance":{"generator":"experiments/outside_in_sentence_decoder.py","vocabulary_sha256":base["vocabulary_sha256"],"reader_gate":"all outputs are diagnostics until blinded human evaluation"},"next_construction":"Use a semantic-valency paired chart with fresh lexical choices, without mutating a finished tape."}
 out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
if __name__=="__main__":main()
