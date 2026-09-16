#!/usr/bin/env python3
"""Preflight nested quotation grammar; blocked on retained reported speech."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/nested-quotation-preflight-20260916.json'
def run():
 return {'experiment_id':'nested-quotation-preflight-20260916','status':'preflight_blocked','requested_signature':'nested-quotation-levels|direct-speech-complement|independent-proposition-realization|reported-speech-mirror-ledger|quotation-repair','overlaps':['reported-speech-topology-20260916','dialogue-speech-act-residual-20260916','synchronous-semantic-parse-equations'],'reason':'Reported-speech topology already retains attribution embedding, tense-shifted content, and live character obligations; dialogue routes retain independent utterance composition. Nested quotation would replay the same reporting topology.','pivot':'Use a non-speech semantic topology such as quantified measurement/scalar comparison with independently lexicalized complete clauses.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
