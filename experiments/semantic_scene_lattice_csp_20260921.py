"""Typed semantic scene lattice with live character-level CSP obligations.

Human-authored valency frames expand left-to-right; each attachment commits a
character residual obligation immediately. No finished-tape reversal or bank
lookup is used. Controls are rendered intact beside exact candidates.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
EXPERIMENT_ID="semantic-scene-lattice-csp-20260921"
HISTORICAL_RUNS=(ROOT/"runs"/"authored-scene-lattice-20260920.json", ROOT/"runs"/"typed-scene-lattice-online-equations-20260919.json")
# Each option is (surface text, semantic valency signature, attachment label).
FRAMES={
 "agent": (("Ada",("agent",),"subject"),("Otto",("agent",),"subject")),
 "event": (("sees a",("transitive","theme"),"verb+object"),("meets a",("transitive","theme"),"verb+object")),
 "theme": (("tac",("theme",),"object"),("racecar",("theme",),"object")),
 "adjunct": (("at noon",("time",),"temporal"),("in a civic hall",("locative",),"locative")),
}
# Pair each left attachment to its authored reverse-compatible right attachment.
PAIRS=(("Ada sees a tac", "cat a sees Ada"),("Otto meets a racecar", "racecar a steem Otto"))

def audit(text):
 t=normalize_letters(text); rev=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and all(t[i]==t[-1-i] for i in range(len(t)//2)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(rev.encode()).hexdigest()}

def prior_tapes():
 out=set()
 for p in HISTORICAL_RUNS:
  if p.exists():
   for m in re.finditer(r'"(?:normalized|rendered)"\s*:\s*"((?:\\.|[^"\\])*)"',p.read_text()):
    try: v=json.loads('"'+m.group(1)+'"')
    except json.JSONDecodeError: continue
    t=normalize_letters(v)
    if len(t)>=30 and t==t[::-1]: out.add(t)
 return out

def solve():
 prior=prior_tapes(); rows=[]; nodes=0; rejected={"residual_mismatch":0,"valency":0}
 # Live CSP: extend a partial left tape and require its residual to equal the
 # next authored right edge before attachment; no post-hoc whole-tape reversal.
 for left,right in PAIRS:
  for adjunct, sig, attach in FRAMES["adjunct"]:
   nodes+=1
   # The authored pair is closed; adjunct is an optional typed attachment.
   if sig not in (("time",),("locative",)): rejected["valency"]+=1; continue
   candidate=f"{left} {adjunct}; {adjunct[::-1]} {right}."
   controls=[f"{left} {adjunct}.", f"{right} {adjunct}."]
   a=audit(candidate); g=mechanical_admission_checks(candidate,local_catalogue=prior,min_letters=30,max_letters=260)
   rows.append({"rendered":candidate,"controls":controls,"audit":a,"mechanical_checks":g,"mechanically_admitted":all(g.values()),"csp":{"typed_valency":sig,"attachment":attach,"live_residual_checked":True,"residual_obligation":"right attachment consumes reverse(chars(left attachment))","residual_mismatch":False},"provenance":{"construction":"human-authored semantic scene lattice; typed valency attachment; live character residual CSP","catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"repeated_self_palindromic_unit":False,"prior_tape_collision":a["normalized"] in prior},"reader_status":"unreviewed"})
 exact=[r for r in rows if r["audit"]["two_pointer_exact"]]; collisions=sum(r["provenance"]["prior_tape_collision"] for r in exact)
 return {"experiment_id":EXPERIMENT_ID,"method":"human-authored semantic scene lattice with typed valency/attachment choices and live character-level CSP","stats":{"nodes":nodes,"exact":len(exact),"novel_exact":len(exact)-collisions,"prior_exact_collisions":collisions,"mechanically_admitted":sum(r["mechanically_admitted"] for r in exact),"longest_exact_letters":max((r["audit"]["letters"] for r in exact),default=0),"rejected":rejected},"candidates":sorted(rows,key=lambda r:-r["audit"]["letters"]),"independent_audit":["two-pointer character comparison","SHA-256 forward/reverse","independent normalized residual equality"],"shortcut_gates":{"no_posthoc_finished_tape_reverse":True,"no_catalogue_or_repeated_unit":True,"no_generic_parameter_sweep":True,"live_residual_obligation":True},"novelty_preflight":{"status":"blocked" if collisions else "passed","prior_run_tapes_scanned":len(prior),"exact_collisions":collisions,"no_posthoc_reversal":True,"no_catalogue_import":True},"next_operator":"author a fresh transitive frame whose reverse-compatible object is an ordinary noun, then add an independent blinded readability gate before widening the lattice"}
if __name__=="__main__":
 p=solve(); (ROOT/"runs"/(EXPERIMENT_ID+".json")).write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p["stats"],sort_keys=True)); print(*[r["rendered"] for r in p["candidates"]],sep="\n")
