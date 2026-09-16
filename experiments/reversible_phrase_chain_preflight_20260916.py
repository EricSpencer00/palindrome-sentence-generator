#!/usr/bin/env python3
"""Preflight reversible phrase-chain graph; blocked on retained chain walks."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reversible-phrase-chain-preflight-20260916.json'
def run():
 return {'experiment_id':'reversible-phrase-chain-preflight-20260916','status':'preflight_blocked','requested_signature':'mirror-pair-phrase-graph|distinct-phrase-path-composition|semantic-transition-constraints|intact-prose-grammar-score|scalable-path-extension','overlaps':['lexical-chain-palindrome-20260916','lexical-chain-walk','connected-role-typed-collocation-path','anaphoric-scene-chain-composition','discourse-graph-walk'],'reason':'The registry already retains typed lexical chains, chain walks, collocation paths, anaphoric scene chains, and discourse graph walks. Using mirror_pairs.json as lexical evidence changes the inventory but not the path-composition state dimension.','pivot':'Use a fresh non-chain operator: conditional embedding with source/evidence state and held-out semantic repair, without mirror-pair lexical evidence.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
