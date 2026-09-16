"""Preflight block for open-vocabulary semantic-evidence grammar."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'open-vocabulary-semantic-evidence-preflight-20260916','status':'preflight_blocked','proposed_signature':'open-vocabulary-grammar|independent-semantic-evidence|character-synchronous-boundary-solving|complete-prose|independent-exact-audit','overlaps':['semantic-selectional-prefix-automaton','heldout-boundary-decoder-20260916','corpus-induced-pos-shapes','open-morpheme-csp-preflight-20260916','grammar-reverse-trie-preflight-20260916'],'reason':'Existing retained families already combine held-out/open vocabulary, semantic selectional evidence, and character-synchronous boundary solving. This proposal changes lexical source/ranking but not the generated linguistic state.','pivot':'No generation run; preserve the block and require a new syntactic or semantic state rather than another vocabulary/decoder wrapper.'}
 (ROOT/'runs/open-vocabulary-semantic-evidence-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
