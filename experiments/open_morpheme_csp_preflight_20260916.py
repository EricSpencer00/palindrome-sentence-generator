"""Preflight block for open-vocabulary morpheme CSP."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'open-morpheme-csp-preflight-20260916','status':'preflight_blocked','proposed_signature':'open-vocabulary-morpheme-csp|anagram-morpheme-choice|inflectional-grammar-validity|scalable-target-length|independent-exact-audit','overlaps':['orthographic-derivational-composition','multiset-balanced-pair-sampling','inflectional-fst-clitic-tape-20260916','wordnet-featured-frame-repair'],'reason':'The registry already contains allomorphic morpheme choice, letter-multiset balance, inflectional transduction, and grammar-valid lexical realization. This proposal would replay those dimensions under an open-vocabulary label.','pivot':'No search run or counted; preserve this block and require an unrepresented linguistic state.'}
 (ROOT/'runs/open-morpheme-csp-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
