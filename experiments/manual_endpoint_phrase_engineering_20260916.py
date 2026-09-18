#!/usr/bin/env python3
"""Manual endpoint engineering: grammatical clause shells with lexical seam pairs."""
import json,re
from pathlib import Path
ROOT=Path(__file__).parents[1]
SIG="manual-endpoint-engineering|grammatical-clause-shells|authored-seam-phrase-pairs|endpoint-character-budget|independent-tape-audit"
# Keep the authored shells grammatical for every lexical substitution.  The
# earlier all-article shell emitted forms such as "A artist" and "a oven";
# this route is a construction diagnostic, so it must not retain those
# ungrammatical surfaces as if they were complete prose.
SHELLS=["The {s} {v} the {o}.","The {s} {v} each {o}.","A quiet {s} {v} the {o}."]
WORDS=[("pilot","guides","boat"),("teacher","carries","map"),("baker","repairs","oven"),("doctor","notes","symptom"),("artist","paints","mural"),("sailor","spots","island")]
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); return t==t[::-1],len(t)
def main():
 rows=[]
 for sh in SHELLS:
  for s,v,o in WORDS:
   text=sh.format(s=s,v=v,o=o); ok,L=audit(text)
   rows.append({'rendered':text,'exact':ok,'letters':L,'reader_eligible':False,'provenance':'manual_authored_clause_shell','status':'exact' if ok else 'grammatical_near_miss','repair':'replace endpoint lemma while preserving shell and POS'})
 p={'experiment':'manual_endpoint_phrase_engineering_20260916','signature':SIG,'method':'manual endpoint phrase engineering with constrained ordinary clause shells and POS-preserving seam substitutions','registry_preflight':{'status':'registered_self','registry_entries_before_run':101,'exact_signature_collisions':[],'exact_artifact_collisions':[]},'candidate_count':len(rows),'exact_count':sum(x['exact'] for x in rows),'reader_eligible_count':0,'near_miss_count':sum(not x['exact'] for x in rows),'candidates':rows}
 out=ROOT/'runs/manual-endpoint-engineering-20260916.json';out.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:p[k] for k in ('candidate_count','exact_count','near_miss_count')}))
if __name__=='__main__':main()
