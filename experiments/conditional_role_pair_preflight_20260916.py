"""Preflight block for conditional role-changing reversible word pairs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'conditional-role-pair-preflight-20260916','status':'preflight_blocked','proposed_signature':'conditional-role-changing-word-pairs|one-sentence-template|heldout-synonym-inflection-repair|joint-character-obligation|independent-audit','overlaps':['whole-sentence-semordnilap-clauses-20260916','typed-semordnilap','wordnet-synonym-frame-csp-20260915','conditional-embedding-solver-20260916','nested-conditional-mutation-20260916'],'reason':'Registry already contains role-changing reversible words, semordnilap clause templates, synonym/inflection repair, and conditional embedding. The proposed route would replay those construction states.','pivot':'No candidate search run or counted; preserve this preflight and select an unused lexical or syntactic state.'}
 (ROOT/'runs/conditional-role-pair-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
