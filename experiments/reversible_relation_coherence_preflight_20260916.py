"""Preflight record for a proposed discourse-planner route.

The proposed center-out discourse planner is blocked as non-novel; this artifact
keeps the evidence and prevents silently counting a replay as a new method.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={"experiment":"reversible-relation-coherence-preflight-20260916","status":"preflight_blocked","proposed_signature":"center-out-discourse-planner|reversible-lexical-relations|topic-continuity|tense-coreference|character-debt-dp|independent-exact-audit","overlaps":["rhetorical-plan-lattice-20260916","discourse-graph-walk-palindrome-20260915","typed-anaphora","centerout-dependency-realizer-20260916"],"reason":"The requested planner state variables and center-out debt search are already represented by retained discourse, anaphora, and center-out dependency families; implementing it would be a relabelled duplicate.","pivot":"No candidate generation was run; retain this preflight as a guard and select a genuinely unused construction dimension next."}
 (ROOT/"runs/reversible-relation-coherence-preflight-20260916.json").write_text(json.dumps(p,indent=2)+"\n")
 print(json.dumps(p))
if __name__=="__main__": main()
