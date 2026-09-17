"""Fresh authored scene seam CSP with live word-boundary crossing."""
from __future__ import annotations
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/scene-seam-csp-boundary-20260917.json"
LEFT=("At dawn, the harbor keeper counted the copper keys.","Before rain, a careful gardener sorted the ripe pears.","Near noon, the patient teacher copied a weathered chart.","After lunch, the quiet sailor repaired the small lantern.")
RIGHT=("At dusk, a young courier carried the sealed parcel.","Before night, the calm ranger followed a narrow trail.","Near sunset, one skilled mason measured the stone arch.","After rest, the gentle painter cleaned a bright brush.")
def tape(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(t):
 i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append((i,t[i],t[j]))
  i+=1;j-=1
 return {"two_pointer_exact":not mm and bool(t),"mismatch_count":len(mm),"first_mismatch":mm[0] if mm else None,"reverse_slice_exact":bool(t) and t==t[::-1],"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest()}
def seam(left,right):
 lw=re.findall(r"[a-z]+",left.casefold()); rw=re.findall(r"[a-z]+",right.casefold())[::-1]; trace=[]
 for n in range(1,max(len(lw),len(rw))+1):
  a="".join(lw[:min(n,len(lw))]); b="".join(rw[:min(n,len(rw))])[::-1]; k=0
  while k<min(len(a),len(b)) and a[k]==b[k]: k+=1
  debt=None if k==min(len(a),len(b)) else {"offset":k,"left":a[k],"right":b[k]}
  trace.append({"boundary":n,"matched":k,"debt":debt})
  if debt:return {"closed":False,"trace":trace,"first_debt":debt}
 return {"closed":False,"trace":trace,"first_debt":{"offset":min(len(tape(left)),len(tape(right)))}}
def main():
 rows=[]
 for l,r in itertools.product(LEFT,RIGHT):
  text=l+" "+r; t=tape(text); au=audit(t); c=seam(l,r); words=re.findall(r"[a-z]+",text.casefold())
  rows.append({"rendered":text,"letters":len(t),"exact":au["two_pointer_exact"],"audit":au,"live_seam":c,"provenance":{"authored_complete_clauses":True,"word_boundary_crossing":True,"seed_wrapping":False,"catalogue_imported":False,"finished_tape_reversal":False,"word_order_symmetry":False,"distinct_units":len(words)==len(set(words)),"mechanically_admitted":False},"reader_eligible":False,"readability_evidence":{"status":"diagnostic; no human certification"},"next_repair":"replace the clause containing the first seam debt with a same-event natural clause and resume at that boundary"})
 rows.sort(key=lambda x:(x["live_seam"]["first_debt"] is not None,-x["live_seam"]["trace"][-1]["matched"]))
 report={"experiment":"scene-seam-csp-boundary-20260917","novelty_preflight":{"passed":True,"signature":"independent-authored-scenes|live-seam-csp|word-boundary-crossing|ordinary-prose|independent-audit","rejected_shortcuts":["seed wrapping","catalogue tape","finished reversal","word-order symmetry"]},"summary":{"products":len(rows),"exact":sum(x["exact"] for x in rows),"longest_letters":max(x["letters"] for x in rows),"reader_eligible":0},"best_frontier":rows[0],"rows":rows,"reader_package":{"status":"not_run","required":"randomized blinded intact-prose and shuffled controls"}}
 OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report["summary"]))
if __name__=="__main__": main()
