"""Preflight block for RhythmAI bank plus reverse grammar decoder."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'rhythmai-open-bank-reverse-decoder-preflight-20260916','status':'preflight_blocked','proposed_signature':'local-model-authored-heldout-bank|open-bank-right-grammar-decoder|reverse-tape-search|complete-prose|independent-exact-audit','overlaps':['rhythmai-authoring-probe-20260916','fixed-tape-gpt2-boundary-decoder-20260915','character-lm-half-tape','model-authored-clause-bank'],'reason':'The registry already contains local model sentence authoring and fixed-tape reverse decoders. Combining these two existing dimensions does not establish a new construction state.','pivot':'No model call or decoder run performed; preserve this preflight and choose an unrepresented linguistic state.'}
 (ROOT/'runs/rhythmai-open-bank-reverse-decoder-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
