#!/usr/bin/env python3
"""Preflight for requested phrase-pair grammar; blocked to avoid replay."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/bidirectional-phrase-pair-preflight-20260916.json'
def run():
 return {'experiment_id':'bidirectional-phrase-pair-preflight-20260916','status':'preflight_blocked','requested_signature':'bidirectional-lexicalized-phrase-pair|semantic-role-frame-joint-authoring|independent-complete-yields|character-mirror-equation|lexical-repair','overlaps':['manual-endpoint-engineering|grammatical-clause-shells|authored-seam-phrase-pairs|endpoint-character-budget|independent-tape-audit','attested-phrase-pair-wrapper|independent-corpus-ngram-index|reverse-tape-segmentation|semantic-composition-around-frozen-center|reader-gated-audit'],'reason':'The requested phrase-pair grammar would replay the retained authored seam-pair and phrase-wrapper construction dimensions; no honest new family remains under this exact framing.','pivot':'Use a new scalable operator: semantic frame expansion with held-out discourse connective states, then independent complete-clause realization and exact tape audit. No candidates were rendered because preflight failed.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
