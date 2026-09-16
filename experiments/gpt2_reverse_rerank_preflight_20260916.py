#!/usr/bin/env python3
"""Preflight GPT-2 proposal + reverse segmentation; blocked on retained neural routes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/gpt2-reverse-rerank-preflight-20260916.json'
def run():
 return {'experiment_id':'gpt2-reverse-rerank-preflight-20260916','status':'preflight_blocked','requested_signature':'local-gpt2-half-proposal|reverse-character-word-segmentation|grammar-reranking|independent-clause-check|boundary-resegmentation-repair','overlaps':['gpt2-topic-half-decoder-20260915','fixed-tape-gpt2-boundary-decoder-20260915','gpt2-byte-pair-token-lattice','neural-dual-prefix-v2'],'reason':'The registry already retains local GPT-2 natural-half proposals, reverse-character decoding, grammar reranking, and boundary/token resegmentation repair. Sampling would replay the same neural proposal plus reverse segmentation dimension.','pivot':'Use a non-neural, non-reverse construction with independently authored semantic operators and a new topology; no GPT-2 samples are admitted as generated evidence.','rendered_candidates':[],'reader_eligible':False}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps({'status':'preflight_blocked','rendered':0}))
