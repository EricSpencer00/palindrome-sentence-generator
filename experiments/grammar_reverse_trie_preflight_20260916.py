"""Preflight block for grammar-coupled reverse-trie construction."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p={'experiment':'grammar-reverse-trie-preflight-20260916','status':'preflight_blocked','proposed_signature':'grammar-coupled-reverse-trie|open-vocabulary-generation|character-obligation|complete-prose|independent-exact-audit','overlaps':['lexical-trie-segmentation-repair-20260916','proper-name-reverse-grammar','semantic-selectional-prefix-automaton','character-trie-product','fixed-tape-gpt2-boundary-decoder-20260915'],'reason':'The registry already contains reverse grammar trie segmentation, character-synchronous trie generation, open-vocabulary boundary search, and fixed-tape decoders. A reverse-trie implementation would alter search mechanics without adding a new linguistic construction state.','pivot':'No generation run; preserve this preflight and require a new state describing what is generated, not how it is ranked or decoded.'}
 (ROOT/'runs/grammar-reverse-trie-preflight-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p))
if __name__=='__main__': main()
